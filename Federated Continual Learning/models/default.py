
import torch.nn as nn
import torch.nn.init as init
# Define your model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(27, 16)
        # init.xavier_uniform_(self.fc1.weight)
        # init.xavier_uniform_(self.fc1.bias)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        # init.xavier_uniform_(self.fc2.weight)
        # init.xavier_uniform_(self.fc2.bias)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        # init.xavier_uniform_(self.fc3.weight)
        # init.xavier_uniform_(self.fc3.bias)



    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        return x


