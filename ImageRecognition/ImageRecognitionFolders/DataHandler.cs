using SkiaSharp; // For image processing (alternative to PIL in Python)
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.transforms;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;
using static TorchSharp.torchvision.transforms;

namespace ImageRecognition.ImageRecognitionFolders
{
    public class TransparentImageDataset : Dataset
    {
        private readonly string _root;
        private readonly List<(string Path, long Label)> _samples;
        private readonly ITransform _imageOnlyTransform;
        private readonly ITransform _maskTransform;

        public TransparentImageDataset(string root
            //, Transform transform = null
            )
        {
            _root = root;

            _samples = LoadSamples(_root);

            // Define image-only transformations
            _imageOnlyTransform = Compose(new ITransform[]
            {
                ColorJitter(brightness: 0.2f, contrast: 0.2f, saturation: 0.2f),
                //Normalize(means: new[] { 0.485d, 0.456d, 0.406d }, stdevs: new[] { 0.229d, 0.224d, 0.225d })
                //Needs resizing?
            });

            // Define mask transformations
            _maskTransform = Compose(new ITransform[]
            {
                Resize(224, 224)
            });
        }

        public override long Count => _samples.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var (path, label) = _samples[(int)index];
            //incorrectly reading 18 and should it be the UUID of the folder

            // Load image and mask
            using var image = SKBitmap.Decode(path);
            var mask = CreateMask(image);

            // Convert to RGB
            var rgbImage = new SKBitmap(image.Width, image.Height);
            using (var canvas = new SKCanvas(rgbImage))
            {
                canvas.DrawBitmap(image, 0, 0);
            }

            // Apply synchronized transformations
            var transformed = ApplySynchronizedTransforms(rgbImage, mask);

            // Convert to tensors
            var imageTensor = ToTensor(transformed.Image);
            var maskTensor = ToTensor(transformed.Mask);

            // Apply image-only transformations
            imageTensor = _imageOnlyTransform.call(imageTensor);

            // Apply mask transformations
            maskTensor = _maskTransform.call(maskTensor);

            // Convert label to tensor
            var labelTensor = torch.tensor(new long[] { label });

            var response = new Dictionary<string, Tensor>
            {
                { "image", imageTensor },
                { "label", labelTensor },
                { "mask", maskTensor }
            };

            return response;
        }

        private static Tensor ToTensor(SKBitmap image)
        {
            // Ensure the image is in the correct format (e.g., RGBA8888)
            if (image.ColorType != SKColorType.Rgba8888)
            {
                image = image.Copy(SKColorType.Rgba8888);
            }

            // Get the pixel data
            var pixels = image.Pixels;

            // Create a tensor to hold the image data
            var tensor = torch.zeros(new long[] { 3, image.Height, image.Width }, dtype: torch.float32);

            // Copy pixel data to the tensor
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = pixels[y * image.Width + x];
                    tensor[0, y, x] = pixel.Red / 255.0f;   // Red channel
                    tensor[1, y, x] = pixel.Green / 255.0f; // Green channel
                    tensor[2, y, x] = pixel.Blue / 255.0f;  // Blue channel
                }
            }

            return tensor;
        }

        private List<(string Path, long Label)> LoadSamples(string root)
        {
            // Load samples from the root directory (similar to ImageFolder in PyTorch)
            var samples = new List<(string Path, long Label)>();
            var classDirs = Directory.GetDirectories(root);
            for (int i = 0; i < classDirs.Length; i++)
            {
                var classDir = classDirs[i];
                var label = i;
                var imagePaths = Directory.GetFiles(classDir);
                foreach (var path in imagePaths)
                {
                    samples.Add((path, label));
                }
            }
            return samples;
        }

        private SKBitmap CreateMask(SKBitmap image)
        {
            var mask = new SKBitmap(image.Width, image.Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var alpha = image.GetPixel(x, y).Alpha;
                    mask.SetPixel(x, y, alpha > 0 ? SKColors.White : SKColors.Black);

                    // maybe these colors should be reversed. White should represent 255 and Black should represent 0.
                }
            }
            return mask;
        }

        private (SKBitmap Image, SKBitmap Mask) ApplySynchronizedTransforms(SKBitmap image, SKBitmap mask)
        {
            var random = new Random();

            // Random horizontal flip
            if (random.NextDouble() > 0.5)
            {
                image = FlipHorizontal(image);
                mask = FlipHorizontal(mask);
            }

            // Random rotation
            if (random.NextDouble() > 0.5)
            {
                var angle = random.Next(-15, 15);
                image = Rotate(image, angle);
                mask = Rotate(mask, angle);
            }

            // Random affine transformation
            if (random.NextDouble() > 0.5)
            {
                var translateX = (float)random.NextDouble() * 0.2f - 0.1f;
                var translateY = (float)random.NextDouble() * 0.2f - 0.1f;
                image = Affine(image, translateX, translateY);
                mask = Affine(mask, translateX, translateY);
            }

            // Random crop
            //if (random.NextDouble() > 0.0)
            //{
            //    var (i, j, h, w) = GetRandomCropParams(image, 112, 112);
            //    image = Crop(image, i, j, h, w);
            //    mask = Crop(mask, i, j, h, w);
            //}

            return (image, mask);
        }

        private SKBitmap FlipHorizontal(SKBitmap bitmap)
        {
            var flipped = new SKBitmap(bitmap.Width, bitmap.Height);
            using (var canvas = new SKCanvas(flipped))
            {
                canvas.Scale(-1, 1, bitmap.Width / 2f, bitmap.Height / 2f);
                canvas.DrawBitmap(bitmap, 0, 0);
            }
            return flipped;
        }

        private SKBitmap Rotate(SKBitmap bitmap, float angle)
        {
            var rotated = new SKBitmap(bitmap.Width, bitmap.Height);
            using (var canvas = new SKCanvas(rotated))
            {
                canvas.RotateDegrees(angle, bitmap.Width / 2f, bitmap.Height / 2f);
                canvas.DrawBitmap(bitmap, 0, 0);
            }
            return rotated;
        }

        private SKBitmap Affine(SKBitmap bitmap, float translateX, float translateY)
        {
            var affine = new SKBitmap(bitmap.Width, bitmap.Height);
            using (var canvas = new SKCanvas(affine))
            {
                canvas.Translate(translateX * bitmap.Width, translateY * bitmap.Height);
                canvas.DrawBitmap(bitmap, 0, 0);
            }
            return affine;
        }

        private (int, int, int, int) GetRandomCropParams(SKBitmap bitmap, int height, int width)
        {
            var i = new Random().Next(0, bitmap.Height - height);
            var j = new Random().Next(0, bitmap.Width - width);
            return (i, j, height, width);
        }

        private SKBitmap Crop(SKBitmap bitmap, int i, int j, int h, int w)
        {
            var cropped = new SKBitmap(w, h);
            using (var canvas = new SKCanvas(cropped))
            {
                canvas.DrawBitmap(bitmap, new SKRect(j, i, j + w, i + h), new SKRect(0, 0, w, h));
            }
            return cropped;
        }

        public class GetTensorResponse
        {
            public Tensor Image { get; set; }
            public Tensor Label { get; set; }
            public Tensor Mask { get; set; }
        }
    }
}
