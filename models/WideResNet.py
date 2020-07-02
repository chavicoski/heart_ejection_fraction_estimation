import sys
sys.path.insert(1, '.')  # To access the libraries
from torch import nn
from torchvision.models import wide_resnet50_2
from models.utils import Flatten

class WideResNet50_0(nn.Module):
    def __init__(self):
        super(WideResNet50_0, self).__init__()
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
    # Test models
    model = WideResNet50_0()
    model.set_freeze(True)
    print(model)
    
