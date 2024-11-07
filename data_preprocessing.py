import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations with data augmentation for training
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),      # Data augmentation
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder('D:\\deep\\Dataset\\Train', transform=transform)
val_dataset = datasets.ImageFolder('D:\\deep\\Dataset\\Validation', transform=transform)
test_dataset = datasets.ImageFolder('D:\\deep\\Dataset\\Test', transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save the DataLoader objects for later use
torch.save({
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader
}, 'D:\\deep\\dataloaders.pth')

print("Data loaders saved.")
