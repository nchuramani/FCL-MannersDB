from torch.utils.data import Dataset

class CustomOutputDataset(Dataset):
    def __init__(self, dataset_pairs):
        self.dataset_pairs = dataset_pairs

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        outputs, labels = self.dataset_pairs[idx]
        return outputs, labels