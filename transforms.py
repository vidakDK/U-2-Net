"""
Tensor Transforms
~~~~~~~~~~~~~~~~~
"""
from skimage import transform as _sktransform
import numpy as np
import torch as _torch


class Resize:
    """ Resize transform """
    def __init__(self, output_size: int):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: np.array) -> np.array:
        new_size = (self.output_size, self.output_size)
        img = _sktransform.resize(image, new_size, mode='constant')
        return img


class ToTensor:
    """ Numpy array to Tensor transform """
    def __call__(self, image: np.array) -> _torch.Tensor:
        """
        Change the r,g,b to b,r,g from [0,255] to [0,1]
        """
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpImg = tmpImg.transpose((2, 0, 1))
        image = _torch.from_numpy(tmpImg)
        return image
