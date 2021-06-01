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
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.Resize((128, 216)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

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

    # Extract Feature extension package
    net = tx.Extractor(net, ["layer4.1.bn2"])

    '''
    for k, v in net.named_parameters():
        print(k, v.requires_grad)
    '''

    # 製作Train LGBM 的資料格式
    dataset = Dogdataset('./dataset/train/', './dataset/meta_train.csv', True, transform=transform_train)
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_dataset, valid_dataset = data.random_split(dataset, [train_set_size, valid_set_size])

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=True, num_workers=4)

    allfeatures = []
    alllabels = []

    with torch.no_grad():
        for data, label in tqdm(train_dataloader):
            data, target = data.cuda(), label.cuda()
            _, features = net(data)
            for f in features['layer4.1.bn2']:
                f = f.reshape(512*4*7).tolist() # Flatten
                allfeatures.append(f)
            for l in label:
                alllabels.append(int(l))

    data = np.array(allfeatures)
    label = np.array(alllabels)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

    param = {'num_leaves': 15, 'objective': 'multiclass', 'num_class': 6, 'max_depth': 10, 'boosting': 'gbdt',
             'learning_rate': 0.01, 'num_threads': 8, 'verbose': 1}

    bst = lgb.train(param, lgb_train, num_boost_round=50, valid_sets=lgb_valid, early_stopping_rounds=5)

    bst.save_model('LGBMModel.txt')


    # 模型加载
    gbm = lgb.Booster(model_file='LGBMModel.txt')
    
    # 模型预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    '''
    model_output, features = net(mfccs.cuda())
    print(model_output)
    feature_shapes = {name: f.shape for name, f in features.items()}
    print(feature_shapes)
    '''


