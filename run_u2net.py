import typing as _t

import numpy as np
import torch as _torch
import torchvision as _torchvision
import transforms as _transforms

from PIL import Image
from skimage import io as _skio
from torch.autograd import Variable

from model import U2NET  # full size version 173.6 MB


class BackgroundRemover:
    def __init__(self, checkpoint_path: str):
        net = U2NET(3, 1)
        net.load_state_dict(_torch.load(checkpoint_path))
        if _torch.cuda.is_available():
            net.cuda()
        net.eval()
        self.net = net

    @staticmethod
    def _to_torch(image: np.array) -> _t.Tuple[_torch.Tensor, tuple]:
        shape = image.shape
        transform = _torchvision.transforms.Compose(
            [_transforms.Resize(320), _transforms.ToTensor()]
        )
        image = transform(image)
        image.unsqueeze_(0)
        image = image.type(_torch.FloatTensor)
        image = Variable(image.cuda())
        return image, shape

    @staticmethod
    def _norm_pred(d):
        """ Normalize the predicted SOD probability map """
        ma = _torch.max(d)
        mi = _torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    @staticmethod
    def _process_result(result: _torch.Tensor, original_shape: tuple,
                        save_image: bool) -> np.array:
        result.squeeze_()
        result_np = result.cpu().data.numpy()
        image = Image.fromarray(result_np * 255).convert('RGB')
        image = image.resize(original_shape[:2], resample=Image.BILINEAR)
        if save_image:
            image.save('./image_without_bg.png')
        return np.asarray(image)

    def process_image(self, image: np.array,
                      save_image: bool = False) -> np.array:
        image, shape = self._to_torch(image)
        d1, d2, d3, d4, d5, d6, d7 = self.net(image)

        # normalization
        pred = d1[:, 0, :, :]
        pred = self._norm_pred(pred)

        result = self._process_result(pred, shape, save_image=save_image)
        return result


if __name__ == "__main__":
    image_path = '~/U-2-Net/test_data/test_images/1_girl.jpg'
    model_dir = './saved_models/u2net/u2net.pth'
    bg_remover = BackgroundRemover(model_dir)
    image = _skio.imread(image_path)
    result = bg_remover.process_image(image, save_image=True)
