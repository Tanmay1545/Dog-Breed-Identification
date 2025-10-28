import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from DogsDataset import DogsDataset
import torch.nn as nn

# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

# Load checkpoint
checkpoint = torch.load("checkpoint.pth.tar", map_location=torch.device('cpu'))
print(f"Checkpoint Keys: {checkpoint.keys()}")

if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
    print("\n✅ Model weights loaded successfully.")
else:
    print("\n❌ No model weights found in checkpoint.")
    exit()

model.eval()

# -----------------------------
# Prepare validation dataset
# -----------------------------
val_dataset = DogsDataset(
    csv_file="dataset/labels/val.csv",
    root_dir="dataset/train/",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Evaluate model accuracy
# -----------------------------
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
