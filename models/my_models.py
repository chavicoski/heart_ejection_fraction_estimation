import torch
from torch import nn

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
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
                nn.Conv2d(64, 128, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
                nn.Conv2d(128, 256, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=0),
                nn.ReLU(),
                #nn.AvgPool2d(())
                nn.Dropout(0.3),
                )
        self.flatten = Flatten()
        self.dense_block = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1)
                )

    def forward(self, x):
        x = self.conv_block(x)
        print(f"Shape pre-flatten: {x.size()}")
        x = self.flatten(x)
        x = self.dense_block(x)
        return x
