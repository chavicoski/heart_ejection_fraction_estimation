import sys
sys.path.insert(1, '.')  # To access the libraries
from torch import nn
from torchvision.models import vgg19
from models.utils import Flatten

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        pretrained_model = vgg19(pretrained=True)

        self.first_conv = nn.Sequential(
                nn.Conv2d(30, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU()
                )
        
        # Get the pretrained convolutional part from VGG19 (avoiding the first layer)
        self.pretrained_block = nn.Sequential(*list(pretrained_model.features.children())[2:])

        self.reduction_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten()
                )

        self.dense_block = nn.Sequential(
                nn.Linear(512, 512),
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
    model = VGG19()
    model.set_freeze(True)
    print(model)
