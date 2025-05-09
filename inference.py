import torch
from src.models.CatDogClassifierV1 import CatDogClassifierV1
from src.models.CatVsDogsV3 import CatsVsDogsV3
from src.loaders.DataLoader import CatVsDogsDataset
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

model1 = CatDogClassifierV1()
model2 = CatsVsDogsV3()

model1.load_state_dict(torch.load(
    "checkpoints/82997207-dd8e-481c-9b08-55f3b4714931.pth"))
model2.load_state_dict(torch.load(
    "checkpoints/42c868e2-1d48-4153-a83a-47c27468eeba_119.pth"))

transform = A.Compose([
    A.Resize(128, 128),
    # Normalize using ImageNet statistics
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB
        if self.transform:
            image = self.transform(image=np.array(image))
            return image['image']
        return image
    


model1.to('cuda')
model2.to('cuda')

# Define the folder containing images
image_folder = "data/inference"  # Replace with the actual path to your image folder

# Create the dataset
dataset = ImageFolderDataset(folder_path=image_folder, transform=transform)

# Create the DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

# Iterate through the DataLoader
for batch_idx, images in enumerate(dataloader):
    images = images.to('cuda')
    output1 = torch.round(torch.sigmoid(model1(images)))
    output2 = torch.round(torch.sigmoid(model2(images)))
    
    
    # Perform inference or other operations on the batch
