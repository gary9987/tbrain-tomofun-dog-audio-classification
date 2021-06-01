import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dogDataset import Dogdataset
import torchvision.transforms as transforms
import torchvision
import torchextractor as tx
import torch.utils.data as data
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import csv

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 216)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    dataset = Dogdataset('./dataset/public_test/', is_train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=40, shuffle=False, num_workers=4)

    #device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet18(pretrained=True)
    num_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(num_features, 6),
        nn.Softmax(1)
    )
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    net.load_state_dict(torch.load('model_CNN.pth'))
    net = tx.Extractor(net, ["layer4.1.bn2"])  # Extract Feature extension package
    net = net.to(device)
    net.eval()

    # 模型加载
    gbm = lgb.Booster(model_file='LGBMModel.txt')



    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other'])
        with torch.no_grad():
            for data, filename in tqdm(dataloader):
                data, target = data.cuda(), data.cuda()
                _, features = net(data)
                flatfeatures = []

                for f in features['layer4.1.bn2']:
                    f = f.reshape(512 * 4 * 7).tolist()  # Flatten
                    flatfeatures.append(f)

                flatfeatures = np.array(flatfeatures)  # list to np array
                # 模型预测
                LBGMPredict = gbm.predict(flatfeatures, num_iteration=gbm.best_iteration)

                for i in range(len(LBGMPredict)):
                    probs = LBGMPredict[i].tolist()
                    writer.writerow([filename[i]]+probs)