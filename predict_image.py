import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

# 🟢 Load breed names from your training CSV
train_csv = pd.read_csv('dataset/labels/train.csv')
breed_names = sorted(train_csv['breed'].unique())

# 🟢 Load the model checkpoint
checkpoint_path = 'checkpoint.pth.tar'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("❌ checkpoint.pth.tar not found! Train your model first.")

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# 🟢 Recreate the same ResNet-18 model architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(breed_names))  # dynamically match number of breeds
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 🟢 Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🟢 Path to your test image (place it in this same folder)
img_path = 'test_image.jpg'  # e.g. 'golden_retriever.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError("❌ test_image.jpg not found! Place a dog image in the project folder.")

# 🟢 Load and preprocess the image
img = Image.open(img_path).convert('RGB')
img_t = transform(img).unsqueeze(0)

# 🧠 Predict the breed
with torch.no_grad():
    outputs = model(img_t)
    _, preds = torch.max(outputs, 1)

predicted_class = preds.item()
predicted_breed = breed_names[predicted_class]

print(f"✅ Predicted Dog Breed: {predicted_breed}")

# 🖼️ Show the image with breed name
plt.imshow(img)
plt.title(f"Predicted Breed: {predicted_breed}")
plt.axis('off')
plt.show()
