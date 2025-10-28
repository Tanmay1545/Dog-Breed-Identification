import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from DogsDataset import DogsDataset  # use your existing dataset class
import torch.nn as nn

# Load the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)  # 120 dog breeds in dataset

# Load saved weights
checkpoint = torch.load("checkpoint.pth.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Validation data setup
val_dataset = DogsDataset('dataset/labels/val.csv', 'dataset/train/', 
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate accuracy
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"\n✅ Validation Accuracy: {val_acc * 100:.2f}%")
