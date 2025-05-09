import torch
from torch import nn

class CatsVsDogsV3(nn.Module):
    def __init__(self, imgChannels=3):
        super(CatsVsDogsV3, self).__init__()
        self.model = nn.Sequential(
            # First Conv Block
            nn.Conv2d(imgChannels, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Fully Connected Layers
            nn.Flatten(),
            nn.Linear(25088, 512),  # Adjust input size based on image dimensions
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Final layer with 1 output node
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    m = CatsVsDogsV3()
    x = torch.randn((32, 3, 128, 128))  # Batch of 32, 3 channels, 128x128 images
    print(m(x).shape)  # Output shape should be (32, 1)
