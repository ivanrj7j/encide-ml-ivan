from torch import nn

class CatDogClassifierV1(nn.Module):
    def __init__(self, imgDim:int, processChannels:int, inputChannels:int=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Sequential(
            nn.Conv2d(inputChannels, processChannels, 3, 2, 1),
            nn.ReLU()
        )

        convs = []
        dim = imgDim//2
        while dim > 1:
            layer = nn.Sequential(
                nn.Conv2d(processChannels, processChannels, 3, 2, 1),
                nn.ReLU()
            )
            convs.append(layer)
            dim //= 2

        self.convs = nn.Sequential(*convs)
        self.output = nn.Sequential(
            nn.Conv2d(processChannels, 1, 1),
            nn.Sigmoid()
        )

        

    def forward(self, x):
        x = self.input(x)
        x = self.convs(x)
        x = self.output(x)
        return x
    

if __name__ == "__main__":
    import torch

    x = torch.randn((64, 4, 128, 128))
    c = CatDogClassifierV1(128, 32, 4)
    out = c(x)
    print(out.shape)