class ResNetClassifier(nn.Module):
    def __init__(self,
                 model_type: str,
                 num_classes: int,
                 freeze: bool = True):
        super(ResNetClassifier, self).__init__()
        # Dictionary to map model type to the corresponding ResNet model and weights
        resnet_models = {
            'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
            'resnet34': (resnet34, ResNet34_Weights.DEFAULT),
            'resnet50': (resnet50, ResNet50_Weights.DEFAULT),
            'resnet101': (resnet101, ResNet101_Weights.DEFAULT),
            'resnet152': (resnet152, ResNet152_Weights.DEFAULT)
        }

        if model_type not in resnet_models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {
                             list(resnet_models.keys())}")

        # Load the ResNet model and weights
        resnet_model, weights = resnet_models[model_type]
        self.resnet = resnet_model(weights=weights)

        # Freeze the ResNet layers except layer4
        for param in self.resnet.parameters():
            param.requires_grad = False

        if not freeze:
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        # Replace the last layer with a new layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x, mask):
        masked_x = x * mask
        return self.resnet(masked_x)
    
def train_resnet(
        model: ResNetClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        criterion: nn.Module,
        num_epochs: int = 10,
        patience: int = 3,
        max_grad_norm: float = 1.0,  # Max norm for gradient clipping
        ) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    gradient_norms = []  # To monitor gradient norms during training

    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels, masks = batch
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            loss.backward()

            # Monitor gradient norms
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)


            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels, masks = batch
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                outputs = model(images, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total * 100

        # Print training and validation stats
        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | '
              f'Val Accuracy: {val_accuracy:.2f}% | Avg Grad Norm: {sum(gradient_norms) / len(gradient_norms):.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save best model state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(val_loss)

    # Load best model weights before returning
    model.load_state_dict(best_model_state)
    return model
# ---------------------------- USAGE ------------------------------------------------
data_dir_path = '../data/trainftn'
dataset = TransparentImageDataset(data_dir_path)

# Assuming a 40 image per class dataset, we get 20 images per class in the training set,
# 15 in the validation set, and 5 in the test set
# These functions return a TransparentImageDataset object
train_dataset, val_dataset = train_val_split(dataset, (1/5))
val_dataset, test_dataset = train_val_split(val_dataset, (1/3))
print(f'Number of classes: {len(train_dataset.classes)}')
# Get the device available for training
device_str ='cuda' if torch.cuda.is_available() else 'cpu'  # Set pin_memory_device correctly

# Set the batch size and create the DataLoader objects
BATCH_SIZE = 64


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True, 
    pin_memory_device=device_str
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True, 
    pin_memory_device=device_str
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True, 
    pin_memory_device=device_str
)

# model = ResNetClassifier(model_type="resnet50", num_classes=len(train_dataset.classes))
model = ResNetClassifier(model_type=model_type, num_classes=len(train_dataset.classes))
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
criterion = CrossEntropyLoss()

# Train the model
trained_model = train_resnet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    num_epochs=30,
    patience=5,
)