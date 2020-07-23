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


class ChannelShift:
    '''Given a 3D tensor (timesteps, height, width) shifts the channels in a circular way'''

    def __init__(self, shift_range=[-0.4, 0.4]):
        '''Constructor
        Params:
            shift_range -> maximum percentaje of channels to shift during transformation in each direction
        '''
        self.shift_range = shift_range

    def __call__(self, x):
        '''Shifts the channels by a randomly picked number of times, depending on the
        sign of the picked number the shift direction changes'''
        shift_size = random.uniform(*self.shift_range)  # Sign denotes direction
        n_shift = round(x.size(0) * shift_size)
        return torch.cat([x[n_shift:], x[:n_shift]])


if __name__ == "__main__":

    import torch
    from matplotlib import pyplot as plt


    print("MyAffine test:")
    tensor = torch.load("../preproc1_150x150_bySlices_dataset_full/train/109_4.pt")
    print(f"orig stats: max: {tensor.max()} - min: {tensor.min()} - mean: {tensor.mean()}")
    transform = MyAffine(angle_range=(-15, 15))
    rot_tensor = transform(tensor)
    print(f"transform stats: max: {rot_tensor.max()} - min: {rot_tensor.min()} - mean: {rot_tensor.mean()}")
    for i in range(30):
        plt.imshow(tensor[i,:,:], cmap=plt.cm.bone)
        plt.savefig(f"test_transform/orig_{i}.png")
        plt.clf()
    for i in range(30):
        plt.imshow(rot_tensor[i,:,:], cmap=plt.cm.bone)
        plt.savefig(f"test_transform/trans_{i}.png")
        plt.clf()


    print("ChannelShift test:")
    dummy = torch.tensor([[[0,0],[0,0]], [[1,1],[1,1]], [[2,2],[2,2]], [[3,3],[3,3]], [[4,4],[4,4]], [[5,5],[5,5]], [[6,6],[6,6]], [[7,7],[7,7]]])
    print(f"dummy orig:\n{dummy}")
    transform = ChannelShift()
    shifted_dummy = transform(dummy)
    print(f"dummy shifted:\n{shifted_dummy}")
