import torch
from torchvision import models
from torch import nn, optim
from Dog_classification import train_model, data_loaders, dataset_sizes

# Load model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

# Load checkpoint
checkpoint = torch.load('checkpoint.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.load_state_dict(checkpoint['optimizer'])

# Continue training for 5–10 more epochs
model = train_model(model, criterion, optimizer, None, dataset_sizes, data_loaders, num_epochs=10)
