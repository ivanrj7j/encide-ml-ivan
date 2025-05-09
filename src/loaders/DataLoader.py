import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CatVsDogsDataset(Dataset):
    def __init__(self, dataDir: str, dim: int = 128):
        """
        Args:
            dataDir (str): Path to the directory containing the data.
        """
        self.dataDir = dataDir
        catImages = list(map(lambda x: (os.path.join(
            dataDir, 'cats', x), 0.0), os.listdir(os.path.join(dataDir, 'cats'))))
        dogImages = list(map(lambda x: (os.path.join(
            dataDir, 'dogs', x), 1.0), os.listdir(os.path.join(dataDir, 'dogs'))))
        self.images = catImages + dogImages
        self.dim = dim
        self.transform = A.Compose([
            A.Resize(dim, dim),  # Resize images to a fixed size
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.RandomRotate90(p=0.5),  # Random 90-degree rotation
            # Small shifts, scaling, and rotations
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            # Adjust brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # Adjust hue, saturation, and value
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            # Adjusted blur_limit to avoid warning
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # Normalize using ImageNet statistics
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image = np.array(image))['image']
        return image, label


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = os.path.join('data', 'test')
    dataset = CatVsDogsDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_data, batch_labels in dataloader:
        print(batch_data.shape, batch_labels)
        break
