import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# Define the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Parameters class
class Params:
    def __init__(self):
        self.batch_size = 16
        self.workers = 4
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 7
        self.lr_gamma = 0.1
        self.epochs = 25
        self.name = "resnet_finetune_vehicles"

params = Params()

# Data directories
data_dir = r'C:\archive'  # Change this path to your folder containing 'train', 'val', and 'test' folders

# Data augmentations and normalization (for training, validation, and testing)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to load test images (no class folders)
def load_test_images(test_dir):
    images = []
    image_paths = []
    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)
        if os.path.isfile(file_path):
            image = Image.open(file_path).convert('RGB')
            image = data_transforms['test'](image)
            images.append(image)
            image_paths.append(file_path)
    return torch.stack(images), image_paths

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=params.epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log the loss and accuracy to TensorBoard
            writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)

        print()

    return model

# Main entry point
if __name__ == '__main__':
    # Load training and validation datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=params.batch_size, shuffle=True, num_workers=params.workers),
        'val': DataLoader(image_datasets['val'], batch_size=params.batch_size, shuffle=False, num_workers=params.workers),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Load a pretrained ResNet model and modify it for our application
    model = models.resnet50(pretrained=True)

    # Freeze all ResNet layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the fully connected layer (ResNet head) for fine-tuning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Move the model to the selected device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)

    # TensorBoard setup
    writer = SummaryWriter(f'runs/{params.name}')

    # Train the model
    model = train_model(model, criterion, optimizer, lr_scheduler)

    # Load and predict on test images
    test_images_dir = r'C:\archive\test'  # Adjust this path as needed
    test_images, test_image_paths = load_test_images(test_images_dir)
    test_images = test_images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        _, preds = torch.max(outputs, 1)

    # Display 10 test images with predictions
    for idx in range(min(10, len(test_image_paths))):  # Limit to 10 images
        image = test_images[idx].cpu().permute(1, 2, 0).numpy()
        image = image * 0.229 + 0.485  # Undo normalization for display
        plt.imshow(image)
        plt.title(f'Predicted Class: {class_names[preds[idx]]}')
        plt.show()

    # Log test accuracy to TensorBoard
    correct = 0
    for idx, pred in enumerate(preds):
        print(f"Image: {os.path.basename(test_image_paths[idx])} -> Predicted Class: {class_names[pred]}")
        correct += 1 if class_names[pred] == "expected_label" else 0  # Replace "expected_label" with your logic
    test_acc = correct / len(preds)
    writer.add_scalar('test accuracy', test_acc)

    # Close TensorBoard writer
    writer.close()
