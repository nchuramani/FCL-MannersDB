
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y
    def __len__(self):
        return len(self.X)

