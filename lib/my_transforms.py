import torchvision.transforms.functional as TF

'''
Here are defined functions to apply data augmentation to pytorch tensors.
This functions can be used like a regular transform from torchvision.transforms
'''

class RandomRotation4D:
    '''Given a 4D tensor (slices, timesteps, height, width) rotates 
    the images inside with the same random angle'''

    def __init__(self, angle_range=[-180, 180]):
        '''Constructor'''
        self.angle_range = angle_range

    def __call__(self, x):
        '''Returns the pytorch tensor "x" with all the images rotated
        with the same angle'''
        return x # TODO
