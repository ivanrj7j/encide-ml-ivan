import torch
from torch import nn

class CatsVsDogsV2(nn.Module):
    def __init__(self):
        super(CatsVsDogsV2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 24 * 24, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer for binary classification
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x    
    
if __name__ == "__main__":
    m = CatsVsDogsV2()
    x = torch.randn((32, 3, 96, 96))
    print(m(x).shape)