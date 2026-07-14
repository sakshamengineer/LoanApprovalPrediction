import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("float32")).view(-1, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]