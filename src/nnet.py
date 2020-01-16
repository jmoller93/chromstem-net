#!/env/python

"""
This is the neural network class

Args:
    For now there are no arguments, but network hyperparameters may be chosen later on

"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv3d(1,16,3)
        self.conv2 = nn.Conv3d(16,32,3)
        self.pool = nn.MaxPool3d(3,3)
        self.fc1   = nn.Linear(32*4*2,100)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

