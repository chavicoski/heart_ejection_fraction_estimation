import sys
sys.path.insert(1, '.')  # To access the libraries
from torch import nn
from torchvision.models import wide_resnet50_2
from models.utils import Flatten

class WideResNet50_0(nn.Module):
    def __init__(self, pretrained=True):
        super(WideResNet50_0, self).__init__()
        pretrained_model = wide_resnet50_2(pretrained=pretrained)
        self.first_conv = nn.Conv2d(30, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pretrained_block = nn.Sequential(*list(pretrained_model.children())[1:7])
        self.reduction_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten()
                )
        self.dense_block = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 1)
                )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.pretrained_block(x)
        x = self.reduction_block(x)
        x = self.dense_block(x)
        return x

    def set_freeze(self, flag):
        '''Sets the requires_grad value to freeze or unfreeze the pretrained part
        of the net'''
        for child in self.pretrained_block.children():
            for param in child.parameters():
                param.requires_grad = not flag
                

class WideResNet50_1(nn.Module):
    def __init__(self, pretrained=True):
        super(WideResNet50_1, self).__init__()
        pretrained_model = wide_resnet50_2(pretrained=pretrained)
        self.first_conv = nn.Sequential(
                nn.Conv2d(30, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.4)
                )
        self.pretrained_block = nn.Sequential(*list(pretrained_model.children())[1:7])
        self.reduction_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten()
                )
        self.dense_block = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 1)
                )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.pretrained_block(x)
        x = self.reduction_block(x)
        x = self.dense_block(x)
        return x

    def set_freeze(self, flag):
        '''Sets the requires_grad value to freeze or unfreeze the pretrained part
        of the net'''
        for child in self.pretrained_block.children():
            for param in child.parameters():
                param.requires_grad = not flag


if __name__ == "__main__":
    # Test models
    model = WideResNet50_0()
    model.set_freeze(True)
    print(model)
    
