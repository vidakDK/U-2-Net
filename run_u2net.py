import os
from skimage import io as _skio, transform as _sktransform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision as _torchvision
from torchvision.transforms import functional as _functional
# from torchvision import transforms
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = _skio.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main(image_path):
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp
    model_dir = './saved_models/u2net/u2net.pth'

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    def load_image(image_path: str):
        image = _skio.imread(image_path)
        # image = _functional.to_tensor(image)
        # image.unsqueeze_(0)
        transform = _torchvision.transforms.Compose(
            [RescaleT(320), ToTensorLab(flag=0)]
        )
        image = transform(image)
        image.unsqueeze_(0)
        image = image.type(torch.FloatTensor)
        image = Variable(image.cuda())
        return image

    def process_image(image, image_path):
        d1, d2, d3, d4, d5, d6, d7 = net(image)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(image_path, pred, 'outputs')

        del d1, d2, d3, d4, d5, d6, d7

    image = load_image(image_path)
    print(f"image shape before processing: {image.shape}")
    process_image(image, image_path)


if __name__ == "__main__":
    image_path = '~/U-2-Net/test_data/test_images/1_girl.jpg'
    main(image_path)
