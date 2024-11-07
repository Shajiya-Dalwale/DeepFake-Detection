from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os

# Define the CNN model structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Added out_channels (128) for conv3
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


# Initialize Flask app
app = Flask(__name__)

# Define upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('D:\\deep\\model.pth'))  # Update the path to your model
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).item()

    return "Fake" if prediction == 0 else "Real"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Ensure the upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Get the prediction result
            result = predict_image(filepath)
            
            return render_template('result.html', filename=file.filename, result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
