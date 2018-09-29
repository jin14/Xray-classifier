import torch
from PIL import Image
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import argparse


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imagedir, labelfile):

        self.imagedir = imagedir
        self.labels = pd.read_csv(labelfile)
        # label disease or not (0 - healthy, 1 - not healthy)
        # self.labels['disease'] = np.where(self.labels['label'] == 10,0,1)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img = os.path.join(self.imagedir, self.labels.iloc[index,0])
        label = self.labels.iloc[index,1]
        return { 'image': transforms.ToTensor()(Image.open(img)),'label':label }

    def __len__(self):
        return len(self.labels)


class BasicConvNet(nn.Module):
    def __init__(self, num_classes):

        super(BasicConvNet, self).__init__()

        #input channels = 3
        # Relu - (3,128,128) to (16,128,128)
        # Pool - (16,128,128) to (16,64,64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Relu - (16,64,64) to (32,64,64)
        # Pool - (32,64,64) to (32,32,32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        #Relu - (32,32,32) to (48,32,32)
        #Pool - (48,32,32) to (48,16,16)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        #Relu - (48,16,16) to (60,16,16)
        #Pool - (60,16,16) to (60,8,8)
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 60, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(8*8*60, num_classes)

        #Relu - (60,8,8) to (120,8,8)
        #Pool - (120,8,8) to (120,4,4)
        self.layer5 = nn.Sequential(
            nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(4*4*120, num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Supply image directory and label filename")
    parser.add_argument('--i', help="File path of original image directory")
    parser.add_argument('--l', help="Label File name")
    args = parser.parse_args()

    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    num_epochs = 10
    num_classes = 15
    batch_size = 100
    learning_rate = 0.001
    validation_split = 0.2
    random_seed = 42

    data = CustomDataset(imagedir = args.i,
                         labelfile = args.l)

    dataSize = data.__len__()
    indices = list(range(int(dataSize)))
    split = int(np.floor(validation_split*dataSize))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=test_sampler)

    model = BasicConvNet(num_classes).to(device)
    criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):
            images = sample['image'].to(device)
            labels = sample['label'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for sample in test_loader:
            images = sample['image'].to(device)
            labels = sample['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_indices), 100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')