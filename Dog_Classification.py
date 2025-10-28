import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import pickle
from skimage import io, transform
from PIL import Image
from DogsDataset import DogsDataset
import csv

# ----------------------------
# Data augmentation & normalization
# ----------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


unorm = UnNormalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))

use_gpu = torch.cuda.is_available()


# ----------------------------
# TRAIN FUNCTION
# ----------------------------
def train():
    datasets_dict = {x: DogsDataset('dataset/labels/' + x + '.csv',
                                    'dataset/train/',
                                    data_transforms[x])
                     for x in ['train', 'val']}

    data_loaders = {x: torch.utils.data.DataLoader(datasets_dict[x],
                                                   batch_size=4,
                                                   shuffle=True,
                                                   num_workers=0)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}

    # Updated pretrained model loading
    model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 120)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, dataset_sizes, data_loaders,
                           num_epochs=25)

    save_checkpoint({'state_dict': model_ft.state_dict()})
    with open('data.pkl', 'wb') as output:
        pickle.dump(model_ft, output)


# ----------------------------
# TRAIN MODEL LOOP
# ----------------------------
def train_model(model, criterion, optimizer, scheduler, dataset_sizes, data_loaders, num_epochs=25, resume=''):
    resume_epoch = 0
    if resume and os.path.isfile(resume):
        print(f"=> loading checkpoint '{resume}'")
        checkpoint = torch.load(resume)
        resume_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
    elif resume:
        print(f"=> no checkpoint found at '{resume}'")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(resume_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            batch_num = 1

            for data in data_loaders[phase]:
                inputs, labels = data
                if batch_num % 100 == 0:
                    print(f"Batch#{batch_num}")
                batch_num += 1

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# ----------------------------
# TEST FUNCTION
# ----------------------------
def test(resume_file):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)
    best_wts = torch.load(resume_file)
    model.load_state_dict(best_wts['state_dict'])

    dataset = DogsDataset(csv_file='test/test_ids.csv',
                          root_dir='test/',
                          transform=data_transforms['val'],
                          mode='test')
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)

    test_model(model, data_loader, len(dataset))


def test_model(model, data_loader, data_size):
    model.train(False)
    with open('predictions.csv', 'w', newline='') as prediction_file:
        csvwriter = csv.writer(prediction_file)
        num_preds = 0
        for data in data_loader:
            if num_preds % 100 == 0:
                print(f'Predictions: {num_preds}/{len(data_loader) - 1}')
                print('-' * 10)

            inputs, ids = data
            if use_gpu:
                inputs = inputs.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            for i in range(len(ids)):
                dog_id = ids[i]
                pred = np.zeros(120)
                pred[preds[i]] = 1
                row = [dog_id] + pred.tolist()
                csvwriter.writerow(row)

            num_preds += 1


# ----------------------------
# VISUALIZATION
# ----------------------------
def visualize_model(model, dataloader, class_names, num_images=6):
    images_so_far = 0

    for data in dataloader:
        inputs, ids = data
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            img_name = os.path.join('test_images/', ids[j] + '.jpg')
            img = Image.open(img_name)
            plt.imshow(img)
            plt.title(f'predicted: {class_names[preds[j]]}')
            print(f'wrote prediction#{images_so_far}')
            plt.savefig(f'predictions/prediction#{images_so_far}.jpg')
            if images_so_far == num_images:
                return


def visualize(resume_file):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    best_wts = torch.load(resume_file)
    model.load_state_dict(best_wts['state_dict'])

    dataset = DogsDataset('test_images/test_ids.csv', 'test_images/',
                          data_transforms['val'], mode='test')
    class_names = pd.read_csv('class_names.csv')['id']

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
    visualize_model(model, data_loader, class_names, num_images=10)


# if __name__ == '__main__':
#     train()
#    if __name__ == '__main__':
#     checkpoint_path = 'checkpoint.pth.tar'

#     # If a checkpoint already exists, resume from it
#     if os.path.exists(checkpoint_path):
#         print(f"✅ Found existing checkpoint: {checkpoint_path}, resuming training...")
#         resume_file = checkpoint_path
#     else:
#         print("🚀 Starting fresh training...")
#         resume_file = ''

#     # Call train_model with resume option
#     train()


if __name__ == '__main__':
    checkpoint_path = 'checkpoint.pth.tar'

    # If a checkpoint already exists, resume from it
    if os.path.exists(checkpoint_path):
        print(f"✅ Found existing checkpoint: {checkpoint_path}, resuming training...")
        resume_file = checkpoint_path
    else:
        print("🚀 Starting fresh training...")
        resume_file = ''

    # Call train_model with resume option
    datasets_dict = {x: DogsDataset('dataset/labels/' + x + '.csv',
                                    'dataset/train/',
                                    data_transforms[x])
                     for x in ['train', 'val']}

    data_loaders = {x: torch.utils.data.DataLoader(datasets_dict[x],
                                                   batch_size=4,
                                                   shuffle=True,
                                                   num_workers=0)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}

    model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 120)
    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, dataset_sizes, data_loaders,
                           num_epochs=25, resume=resume_file)
