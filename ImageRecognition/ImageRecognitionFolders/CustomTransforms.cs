using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace ImageRecognition.ImageRecognitionFolders
{
    public enum ColorChannel
    {
        Red,
        Green,
        Blue,
        Alpha
    }

    public static class CustomTransforms
    {
        public static Tensor ToTensor(SKBitmap image)
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

        //public static SKBitmap GetChannel(this SKBitmap image, ColorChannel channel)
        //{
        //    // Implement logic to extract a specific channel from the image
        //    return image;
        //}

        //public static SKBitmap ConvertToMask(this SKBitmap alphaChannel)
        //{
        //    // Implement logic to convert alpha channel to a binary mask
        //    return alphaChannel;
        //}

        //public static SKBitmap GetRgb(this SKBitmap image)
        //{
        //    // Convert the image to RGB format (remove alpha channel)
        //    if (image.ColorType == SKColorType.Rgba8888)
        //    {
        //        var rgbImage = new SKBitmap(image.Width, image.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);
        //        using (var canvas = new SKCanvas(rgbImage))
        //        {
        //            canvas.DrawBitmap(image, 0, 0);
        //        }
        //        return rgbImage;
        //    }
        //    return image;
        //}

        //public static SKBitmap FlipHorizontal(this SKBitmap image)
        //{
        //    // Implement logic to flip the image horizontally
        //    var flipped = new SKBitmap(image.Width, image.Height);
        //    using (var canvas = new SKCanvas(flipped))
        //    {
        //        canvas.Scale(-1, 1, image.Width / 2f, image.Height / 2f);
        //        canvas.DrawBitmap(image, 0, 0);
        //    }
        //    return flipped;
        //}

        //public static SKBitmap Rotate(this SKBitmap image, float angle)
        //{
        //    // Implement logic to rotate the image
        //    var rotated = new SKBitmap(image.Width, image.Height);
        //    using (var canvas = new SKCanvas(rotated))
        //    {
        //        canvas.RotateDegrees(angle, image.Width / 2f, image.Height / 2f);
        //        canvas.DrawBitmap(image, 0, 0);
        //    }
        //    return rotated;
        //}

        //public static SKBitmap Affine(this SKBitmap image, float angle, (double, double) translate, double scale, float shear)
        //{
        //    // Implement logic to apply affine transformation
        //    var affine = new SKBitmap(image.Width, image.Height);
        //    using (var canvas = new SKCanvas(affine))
        //    {
        //        var matrix = SKMatrix.CreateIdentity();
        //        matrix = matrix.PostConcat(SKMatrix.CreateTranslation((float)translate.Item1, (float)translate.Item2));
        //        matrix = matrix.PostConcat(SKMatrix.CreateScale((float)scale, (float)scale));
        //        matrix = matrix.PostConcat(SKMatrix.CreateRotationDegrees(angle));
        //        canvas.SetMatrix(matrix);
        //        canvas.DrawBitmap(image, 0, 0);
        //    }
        //    return affine;
        //}

        //public static SKBitmap Crop(this SKBitmap image, (int, int, int, int) cropParams)
        //{
        //    // Implement logic to crop the image
        //    var (x, y, width, height) = cropParams;
        //    return new SKBitmap(image).ExtractSubset(new SKRectI(x, y, x + width, y + height));
        //}
    }
}
