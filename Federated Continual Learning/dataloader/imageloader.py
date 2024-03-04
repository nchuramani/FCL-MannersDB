from torch.utils.data import Dataset
import torch
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        image = image.crop((295, 0, 295+1018, image.size[1]))
    
        labels = torch.tensor(self.dataframe.iloc[idx, 4:].values.astype('float32'), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels