import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import math
from utils import *
import cv2
from scipy import misc
from PIL import Image

class MattingDataSet(Dataset):
    def __init__(self, fg_path, a_path, bg_path, merged_path):
        self.fg_path = fg_path
        self.a_path = a_path
        self.bg_path = bg_path
        self.merged_path = merged_path
        self.merged_lst = os.listdir(self.merged_path)

    def __len__(self):

        return len(self.merged_lst)

    def __getitem__(self, item):
        #Load Data from Disk
        img_name, _ = self.merged_lst[item].split('.')
        fg_name, bg_name = img_name.split('+')
        img = Image.open(self.merged_path + img_name + '.png')
        fg = Image.open(self.fg_path + fg_name + '.jpg')
        bg = Image.open(self.bg_path + bg_name + '.jpg')
        alpha = Image.open(self.a_path + fg_name + '.png')

        #Convert to RGB if not
        if bg.mode != 'RGB' and bg.mode != 'RGBA':
            bg = bg.convert('RGB')
        if img.mode != 'RGB' and img.mode != 'RGBA':
            img = img.convert('RGB')

        #if bg is smaller than fg, resize bg to assert bg.size >= fg.size
        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        w, h = img.size
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = bg.resize((math.ceil(bw*ratio),math.ceil(bh*ratio)), Image.BICUBIC)


        #Convert the PIL.Image format to numpy.array
        fg = np.array(fg)
        bg = np.array(bg)
        alpha = np.array(alpha)
        trimap = generate_trimap(alpha)
        img = np.array(img)

        #crop the images
        ##get the location of the bounding box
        loc = random_sample(img, trimap)
        img_patch = data_crop(img, loc)
        fg_patch = data_crop(fg, loc)
        trimap_patch = data_crop(trimap, loc)
        alpha_patch = data_crop(alpha, loc)
        bg_patch = data_crop(bg, loc)

        #generate the bool matrix to locate the unknown region note that trimap has not / 255.0
        unknown = binary_unknown(trimap_patch)[np.newaxis, :, :]

        ## covert the dim to the right order and right type

        img_patch = img_patch.transpose(2, 0, 1) / 255.0
        alpha_patch = alpha_patch[np.newaxis, :, :] / 255.0
        trimap_patch = trimap_patch[np.newaxis, :, :] / 255.0
        fg_patch = fg_patch.transpose(2, 0, 1) / 255.0
        bg_patch = bg_patch.transpose(2, 0, 1) / 255.0


        return {'img': img_patch, 'fg': fg_patch, 'bg': bg_patch, 'trimap': trimap_patch, 'alpha': alpha_patch, 'unknown':unknown}





