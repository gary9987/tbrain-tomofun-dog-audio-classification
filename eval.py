import csv
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dogDataset import Dogdataset
from model import Model

import torchvision.transforms as transforms
import torchvision

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 79)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    dataset = Dogdataset('./dataset/public_test/', is_train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet18(pretrained=True)
    num_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 6),
        nn.Softmax(1)
    )
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    net.load_state_dict(torch.load('model_CNN.pth'))
    net = net.to(device)
    net.eval()

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other'])
        with torch.no_grad():
            for data, filename in tqdm(dataloader):
                data, target = data.cuda(), data.cuda()
                output = net(data)

                output = output.tolist()[0]
                #print(filename[0]+out)
                writer.writerow([filename[0]]+output)

