using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ImageRecognition.ImageRecognitionFolders.ResNetClassifier
{
    public class ResNetClassifierModel : Module
    {
        private readonly Module<Tensor, Tensor> resnet;
        private readonly Linear fc;

        public ResNetClassifierModel(string modelType, int numClasses, bool freeze = true) : base("ResNetClassifier")
        {
            // Dictionary to map model type to the corresponding ResNet model and weights
            var resnetModels = new Dictionary<string, Func<Module<Tensor, Tensor>>>
        {
            //{ "resnet18", () => torchvision.models.resnet18(pretrained: true) },
            //{ "resnet34", () => torchvision.models.resnet34(pretrained: true) },
            //{ "resnet50", () => torchvision.models.resnet50(pretrained: true) },
            //{ "resnet101", () => torchvision.models.resnet101(pretrained: true) },
            //{ "resnet152", () => torchvision.models.resnet152(pretrained: true) }
            { "resnet50", () => torchvision.models.resnet50( weights_file: "ImageRecognition/ImageRecognitionFolders/Weights/resnet50_default_weights.dat" ) }
        };

            if (!resnetModels.ContainsKey(modelType))
            {
                throw new ArgumentException($"Unsupported model type: {modelType}. Supported types are: {string.Join(", ", resnetModels.Keys)}");
            }

            // Load the ResNet model
            resnet = resnetModels[modelType]();

            // Freeze the ResNet layers except layer4
            if (freeze)
            {
                foreach (var param in resnet.parameters())
                {
                    param.requires_grad = false;
                }

                // Unfreeze layer4
                foreach (var param in resnet.named_parameters())
                {
                    if (param.name.StartsWith("layer4"))
                    {
                        param.parameter.requires_grad = true;
                    }
                }
            }

            // Replace the last layer with a new layer
            var inFeatures = ((Linear)resnet.named_children().Last().module).in_features;
            fc = Linear(inFeatures, numClasses);
            resnet.register_module("fc", fc);
        }

        public
            //override
            Tensor forward(Tensor x, Tensor mask)
        {
            var masked_x = x * mask;
            return resnet.forward(masked_x);
        }
    }
}
