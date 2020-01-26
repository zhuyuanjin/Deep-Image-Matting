import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import math
from utils import *

from scipy import misc
from PIL import Image

class MattingDataSet(Dataset):
    def __init__(self, img_path, a_path):
        self.img_path = img_path
        self.a_path = a_path
        self.img_lst = os.listdir(self.img_path)	

    def __len__(self):

        return 1000

    def __getitem__(self, item):
        img_name, _ = self.img_lst[item].split(".")
        alpha = np.array(Image.open(os.path.join(self.a_path, img_name+'.png')))
        img = np.array(Image.open(os.path.join(self.img_path, img_name+'.jpg')))
        _kernel_size = np.random.randint(3,8,1).item()
        trimap = generate_trimap(alpha, (_kernel_size, _kernel_size))
        loc = random_sample(img, trimap)
        img_patch = data_crop(img, loc)
        trimap_patch = data_crop(trimap, loc)
        alpha_patch = data_crop(alpha, loc)
        unknown = binary_unknown(trimap_patch)[np.newaxis, :, :]
        img_patch = img_patch.transpose(2, 0, 1) / 255.0
        alpha_patch = alpha_patch[np.newaxis, :, :] / 255.0
        trimap_patch = trimap_patch[np.newaxis, :, :] / 255.0
        return {'img': img_patch,   'trimap': trimap_patch, 'alpha': alpha_patch, 'unknown':unknown}






