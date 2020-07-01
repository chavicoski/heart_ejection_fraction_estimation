import torch
from torch import nn
from torchvision.models import wide_resnet50_2

class Flatten(nn.Module):
    '''Auxiliary module to do flatten operation'''
    def forward(self, x):
        return x.view(x.size(0), -1)

class Time_as_depth_model(nn.Module):
    '''
    Convolutional model that takes as input a tensor of shape (batch_size, timesteps, H, W)
    and uses the 'timesteps' dimension as the channels to employ 2D convolutions. The model
    outputs a single value to make regresion with the systole or diastole value.
    '''
    def __init__(self, in_channels=30):
        super(Time_as_depth_model, self).__init__()
        # Define the model architecture
        self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.4),
                nn.Conv2d(64, 128, kernel_size=3, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.4),
                nn.Conv2d(128, 256, kernel_size=3, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.AvgPool2d((11, 11))
                )
        self.flatten = Flatten()
        self.dense_block = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 1)
                )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        return x


class WideResNet50(nn.Module):
    def __init__(self):
        super(WideResNet50, self).__init__()
        pretrained_model = wide_resnet50_2(pretrained=True)
        self.first_conv = nn.Conv2d(30, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.pretrained_block = nn.Sequential(*list(pretrained_model.children())[1:7])
        self.reduction_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten()
                )
        self.dense_block = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
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
    model = WideResNet50()
    model.set_freeze(True)
    print(model)
    
