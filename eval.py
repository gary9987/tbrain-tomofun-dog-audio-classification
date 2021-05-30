import csv
import glob
import librosa
from os.path import basename
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dogDataset import Dogdataset
from model import Model
import numpy as np

if __name__ == '__main__':

    dataset = Dogdataset('./dataset/public_test/', is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = 'cpu'
    net = Model(128, 6)

    net.load_state_dict(torch.load('model_CNN.pth'))
    net = net.to(device)
    net.eval()

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other'])
        with torch.no_grad():
            for input, filename in tqdm(dataloader):
                input.to(device)
                output = net(input)
                output = output.tolist()[0]
                #print(filename[0]+out)
                writer.writerow([filename[0]]+output)

