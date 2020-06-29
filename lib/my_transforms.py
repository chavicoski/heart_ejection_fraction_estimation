import torch
import torchvision.transforms.functional as TF
import random
import PIL

'''
Here are defined functions to apply data augmentation to pytorch tensors.
This functions can be used like a regular transform from torchvision.transforms
'''

class MyAffine:
    '''Given a 3D tensor (timesteps, height, width) rotates and translates 
    all the images from each timestep with the same random value'''

    def __init__(self, angle_range=(-180, 180), translate_range=(0, 0.1)):
        '''Constructor
        Params:
            angle_range: Float range to take the random angle for rotation
            translate_range: Float range to take the percenaje of pixels from
                             each dimension to make the translation
        '''
        self.angle_range = angle_range
        self.translate_range = translate_range

    def __call__(self, x):
        '''Returns a pytorch tensor with the rotation and translation applied to
        all the channels with the same random values'''
        channels, h, w = x.size()
        # Compute random parameters
        rot_angle = random.uniform(*self.angle_range)  # Get random angle
        trans_v = int(h * random.uniform(*self.translate_range)) # Vertical pixels
        trans_h = int(w * random.uniform(*self.translate_range)) # Horizontal pixels
        res = torch.zeros((channels, h, w))  # To store result
        for ch in range(channels):
            ch_data = x[ch,:,:]
            ch_pil = TF.to_pil_image(ch_data)
            ch_pil = TF.affine(ch_pil, rot_angle, (trans_v, trans_h), 1, 0, resample=PIL.Image.BICUBIC)
            res[ch,:,:] = TF.to_tensor(ch_pil)
        return res


if __name__ == "__main__":

    import torch
    from matplotlib import pyplot as plt

    tensor = torch.load("../preproc1_150x150_bySlices_dataset/train/109_4.pt")
    transform = MyAffine(angle_range=(-15, 15))
    rot_tensor = transform(tensor)
    for i in range(30):
        plt.imshow(tensor[i,:,:], cmap=plt.cm.bone)
        plt.savefig(f"test_transform/orig_{i}.png")
        plt.clf()
    for i in range(30):
        plt.imshow(rot_tensor[i,:,:], cmap=plt.cm.bone)
        plt.savefig(f"test_transform/trans_{i}.png")
        plt.clf()

