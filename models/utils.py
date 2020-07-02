from torch import nn

class Flatten(nn.Module):
    '''Auxiliary module to do flatten operation'''
    def forward(self, x):
        return x.view(x.size(0), -1)
