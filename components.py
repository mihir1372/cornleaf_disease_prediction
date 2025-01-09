import torch
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Dropout,
    Softmax,
)


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2)
        )
        
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        
        self.classifier = Sequential(
            Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=4, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)