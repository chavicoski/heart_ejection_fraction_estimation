import sys
sys.path.insert(1, '.')  # To access the libraries
from torch import nn
from torchvision.models import densenet121
from models.utils import Flatten

class DenseNet121_0(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet121_0, self).__init__()
        pretrained_model = densenet121(pretrained=pretrained)
        self.first_conv = nn.Sequential( 
                nn.Conv2d(30, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.pretrained_block = nn.Sequential(*list(pretrained_model.features.children())[4:])
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


if __name__ == "__main__":
    # Test models
    model = DenseNet121_0()
    model.set_freeze(True)
    print(model)
