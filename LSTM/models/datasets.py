from torch.utils.data import Dataset, DataLoader

class PMUDataset(Dataset):
    def __init__(self, data):
        self.data = data.float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


