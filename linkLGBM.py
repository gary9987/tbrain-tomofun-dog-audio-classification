import csv
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dogDataset import Dogdataset
from model import Model

import torchvision.transforms as transforms
import torchvision
import torchextractor as tx

if __name__ == '__main__':

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet18(pretrained=False)
    num_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(num_features, 6),
        nn.Softmax(1)
    )
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    net.load_state_dict(torch.load('model_CNN.pth'))
    net = net.to(device)
    net.eval()

    model = tx.Extractor(net, ["layer4"])
