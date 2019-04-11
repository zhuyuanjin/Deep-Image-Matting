import torch
import numpy as np
import os
import cv2
import random
from scipy import misc



def data_crop(data, loc=(0, 0), crop_size=(320, 320)):
    max_w, max_h = data.shape[0], data.shape[1]
    height, width = crop_size
    x, y = loc
    up = int (y - height / 2)
    down = int(y + height / 2)
    left = int(x - width / 2)
    right = int(x + width / 2)
    if up < 0 or left < 0 or down > max_h or right > max_w:
        raise IndexError
    patch = data[left:right, up:down]

    return patch



def generate_trimap(alpha, kernel_size=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)



def random_sample(data, trimap, crop_size=(320, 320)):
    try:
        w, h = data.shape[0], data.shape[1]
    except:
        print(type(data))
    unknown_loc = np.array(list(zip(*np.where(trimap == 128))))
    n_unkown = len(unknown_loc)
    i = 1
    while i<100:
        index = np.random.randint(0, n_unkown)
        loc = unknown_loc[index]
        height, width = crop_size
        x, y = loc
        up = int(y - height / 2)
        down = int(y + height / 2)
        left = int(x - width / 2)
        right = int(x + width / 2)
        if up < 0 or left < 0 or down > h or right > w:
            continue
        else:
            return loc
    return (int(w/2), int(h/2))




def composite4(fg, bg, a, w, h):
    bbox = fg.getbbox()
    bg = bg.crop((0, 0, w, h))

    fg_list = fg.load()
    bg_list = bg.load()
    a_list = a.load()

    for y in range(h):
        for x in range(w):
            alpha = a_list[x, y] / 255.0
            t = fg_list[x, y][0]
            t2 = bg_list[x, y][0]
            if alpha >= 1:
                r = int(fg_list[x, y][0])
                g = int(fg_list[x, y][1])
                b = int(fg_list[x, y][2])
                bg_list[x, y] = (r, g, b, 255)
            elif alpha > 0:
                r = int(alpha * fg_list[x, y][0] + (1 - alpha) * bg_list[x, y][0])
                g = int(alpha * fg_list[x, y][1] + (1 - alpha) * bg_list[x, y][1])
                b = int(alpha * fg_list[x, y][2] + (1 - alpha) * bg_list[x, y][2])
                bg_list[x, y] = (r, g, b, 255)

    return bg

def binary_unknown(trimap):
    return np.array(trimap == 128, dtype=np.float)

