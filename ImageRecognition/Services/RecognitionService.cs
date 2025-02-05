using ImageRecognition.ImageRecognitionFolders;
using ImageRecognition.ImageRecognitionFolders.ResNetClassifier;
using TorchSharp;
using TorchSharp.Modules;

namespace ImageRecognition.Services
{
    public interface IRecognitionService
    {
        ResNetClassifierModel GetResNetClassifierModel();
    }

    public class RecognitionService : IRecognitionService
    {
        public ResNetClassifierModel GetResNetClassifierModel()
        {
            string dataPath = "objects";

            var dataset = new TransparentImageDataset(dataPath);

            string deviceString = torch.cuda.is_available() ? "cuda" : "cpu";

            int batchSize = 64;

            var trainLoader = new DataLoader(dataset, batchSize, shuffle: true);

            var model = new ResNetClassifierModel("resnet50", trainLoader.Count());

            return new ResNetClassifierModel("resnet50", 2);
        }
    }
}
