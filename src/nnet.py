#!/env/python

"""
This is the neural network class

Args:
    For now there are no arguments, but network hyperparameters may be chosen later on

"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, filter_size, pool_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, filter_size)
        self.conv2 = nn.Conv3d(8, 12, filter_size)
        self.pool = nn.MaxPool3d(pool_size, pool_size)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(12 * 7 ** 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
