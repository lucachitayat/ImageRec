using ImageRecognition.ImageRecognitionFolders.ResNetClassifier;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ImageRecognition.ImageRecognitionFolders
{
    public class ModelTrainerService
    {
        private readonly DataLoader _trainLoader;
        private readonly DataLoader _valLoader;
        private readonly Optimizer _optimizer;
        private readonly LRScheduler _scheduler;
        private readonly Loss<torch.Tensor, torch.Tensor, torch.Tensor> _criterion;

        public ModelTrainerService(
            DataLoader trainLoader,
            DataLoader valLoader,
            Optimizer optimizer,
            LRScheduler scheduler,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion)
        {
            _trainLoader = trainLoader;
            _valLoader = valLoader;
            _optimizer = optimizer;
            _scheduler = scheduler;
            _criterion = criterion;
        }

        public static Module TrainResNet(
            ResNetClassifierModel model,
            DataLoader trainLoader,
            DataLoader valLoader,
            Optimizer optimizer,
            LRScheduler scheduler,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion,
            int numEpochs = 10,
            int patience = 3,
            float maxGradNorm = 1.0f)
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            model.to(device);

            var bestModelState = model.state_dict();
            var bestValLoss = float.PositiveInfinity;
            var patienceCounter = 0;
            var gradientNorms = new List<float>();

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                // Training phase
                model.train();
                float trainLoss = 0;

                //foreach (var (images, labels, masks) in trainLoader)
                //{
                foreach (var batch in trainLoader)
                {
                    var images = batch["images"];
                    var labels = batch["labels"];
                    var masks = batch["masks"];

                    images = images.to(device);
                    labels = labels.to(device);
                    masks = masks.to(device);

                    optimizer.zero_grad();
                    var outputs = model.forward(images, masks);
                    var loss = criterion.forward(outputs, labels);
                    loss.backward();

                    // Gradient clipping
                    var totalNorm = 0.0f;
                    foreach (var param in model.parameters())
                    {
                        if (param.grad is not null)
                        {
                            var paramNorm = param.grad.norm(2).item<float>();
                            totalNorm += paramNorm * paramNorm;
                        }
                    }
                    totalNorm = (float)Math.Sqrt(totalNorm);
                    gradientNorms.Add(totalNorm);

                    if (totalNorm > maxGradNorm)
                    {
                        foreach (var param in model.parameters())
                        {
                            if (param.grad is not null)
                            {
                                param.grad.mul_(maxGradNorm / (totalNorm + 1e-6));
                            }
                        }
                    }

                    optimizer.step();
                    trainLoss += loss.item<float>();
                }

                trainLoss /= trainLoader.Count;

                // Validation phase
                model.eval();
                float valLoss = 0;
                int correct = 0;
                int total = 0;

                using (torch.no_grad())
                {
                    //foreach (var (images, labels, masks) in valLoader)
                    //{
                    foreach (var batch in trainLoader)
                    {
                        var images = batch["images"];
                        var labels = batch["labels"];
                        var masks = batch["masks"];

                        images = images.to(device);
                        labels = labels.to(device);
                        masks = masks.to(device);

                        var outputs = model.forward(images, masks);
                        var loss = criterion.forward(outputs, labels);
                        valLoss += loss.item<float>();

                        // Compute accuracy
                        var predicted = outputs.argmax(1);
                        correct += (predicted == labels).sum().item<int>();
                        //total += labels.size(0);
                        total += (int)labels.size(0);
                    }
                }

                valLoss /= valLoader.Count;
                var valAccuracy = 100.0f * correct / total;

                // Print training and validation stats
                Console.WriteLine($"Epoch [{epoch + 1}/{numEpochs}] | " +
                                  $"Train Loss: {trainLoss:F5} | Val Loss: {valLoss:F5} | " +
                                  $"Val Accuracy: {valAccuracy:F2}% | Avg Grad Norm: {gradientNorms.Average():F4}");

                // Early stopping
                if (valLoss < bestValLoss)
                {
                    bestValLoss = valLoss;
                    bestModelState = model.state_dict();
                    patienceCounter = 0;
                }
                else
                {
                    patienceCounter++;
                    if (patienceCounter >= patience)
                    {
                        Console.WriteLine($"Early stopping at epoch {epoch + 1}");
                        break;
                    }
                }

                // Step the scheduler
                scheduler.step(valLoss);
            }

            // Load best model weights
            model.load_state_dict(bestModelState);

            return model;
        }

        //public static void ValidateModel(Module model, DataLoader valLoader, Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion)
        //{
        //}
    }
}
