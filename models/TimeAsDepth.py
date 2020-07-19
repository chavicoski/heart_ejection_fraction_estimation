import sys
sys.path.insert(1, '.')  # To access the libraries
from torch import nn
from models.utils import Flatten

class TimeAsDepth_0(nn.Module):
    '''
    Convolutional model that takes as input a tensor of shape (batch_size, timesteps, H, W)
    and uses the 'timesteps' dimension as the channels to employ 2D convolutions. The model
    outputs a single value to make regresion with the systole or diastole value.
    '''
    def __init__(self, in_channels=30):
        super(TimeAsDepth_0, self).__init__()
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


class TimeAsDepth_1(nn.Module):
    '''
    Convolutional model that takes as input a tensor of shape (batch_size, timesteps, H, W)
    and uses the 'timesteps' dimension as the channels to employ 2D convolutions. The model
    outputs a single value to make regresion with the systole or diastole value.
    '''
    def __init__(self, in_channels=30):
        super(TimeAsDepth_1, self).__init__()
        # Define the model architecture
        self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=0),
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


class TimeAsDepth_2(nn.Module):
    '''
    Convolutional model that takes as input a tensor of shape (batch_size, timesteps, H, W)
    and uses the 'timesteps' dimension as the channels to employ 2D convolutions. The model
    outputs a single value to make regresion with the systole or diastole value.
    '''
    def __init__(self, in_channels=30):
        super(TimeAsDepth_2, self).__init__()
        # Define the model architecture
        self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=0),
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


