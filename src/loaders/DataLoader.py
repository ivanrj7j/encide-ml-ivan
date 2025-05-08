import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class CatVsDogsDataset(Dataset):
    def __init__(self, dataDir:str):
        """
        Args:
            dataDir (str): Path to the directory containing the data.
        """
        self.dataDir = dataDir
        catImages = list(map(lambda x: (os.path.join(dataDir, 'cats', x), 0), os.listdir(os.path.join(dataDir, 'cats'))))
        dogImages = list(map(lambda x: (os.path.join(dataDir, 'dogs', x), 1), os.listdir(os.path.join(dataDir, 'dogs'))))
        self.images = catImages + dogImages

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert('RGB')
        image = torch.tensor((np.array(image) / 255.0).transpose(2, 0, 1), dtype=torch.float32)
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        return image, label

# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    data_dir = os.path.join('data', 'train')
    dataset = CatVsDogsDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_data, batch_labels in dataloader:
        print(batch_data.shape, batch_labels)
        break