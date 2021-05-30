import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_size),
            nn.Softmax()
        )
    def forward(self, x_batch):
        return self.network(x_batch)