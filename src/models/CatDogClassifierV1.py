from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class CatDogClassifierV1(nn.Module):
    def __init__(self):
        super(CatDogClassifierV1, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Initial convolution
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 5 Residual Blocks
        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.resblock3 = ResidualBlock(128, 256, stride=2)
        self.resblock4 = ResidualBlock(256, 512, stride=2)
        self.resblock5 = ResidualBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512, 1)  # Fully connected layer for binary classification

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.randn((64, 3, 128, 128))  # Example input
    model = CatDogClassifierV1()
    out = model(x)
    print(out.shape)  # Should output (64, 1)
