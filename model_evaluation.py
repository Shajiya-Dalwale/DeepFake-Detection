import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from collections import Counter

# Define the CNN model structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('D:\\deep\\model.pth'))
model.eval()

# Count images in each class for dataset verification
train_dir = 'D:\\deep\\Dataset\\Train'
class_counts = Counter(os.listdir(train_dir))

for class_name, count in class_counts.items():
    print(f'{class_name}: {count} images')

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).item()
        print(f'Raw output: {output.item()}')  # Print raw output

    return "Fake" if prediction == 0 else "Real"

# Example usage
image_path = 'D:\\deep\\Dataset\\Test\\Real\\real_4.jpg'  # Update with your image path
result = predict_image(image_path)
print(f'The image is predicted to be: {result}')
