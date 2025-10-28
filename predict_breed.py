import torch
from torchvision import models, transforms
from PIL import Image
import json

# Load the trained model checkpoint
checkpoint = torch.load('checkpoint.pth.tar', map_location='cpu')

# Define the model architecture (same as training)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 120)  # 120 dog breeds
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Define same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load a test image
img_path = "test_image.jpg"  # 👈 replace with your image name
img = Image.open(img_path)
img_t = transform(img).unsqueeze(0)

# Predict the breed
with torch.no_grad():
    outputs = model(img_t)
    _, preds = torch.max(outputs, 1)

# If you have breed names mapping file
try:
    with open('breed_to_idx.json') as f:
        breed_to_idx = json.load(f)
    idx_to_breed = {v: k for k, v in breed_to_idx.items()}
    print("Predicted breed:", idx_to_breed[preds.item()])
except:
    print("Predicted class index:", preds.item())
