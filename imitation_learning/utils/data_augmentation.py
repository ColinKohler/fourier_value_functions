import numpy as np
import numpy.random as npr

def random_crop(img, out_size):
    T, C, H, W = img.shape

    x = npr.randint(0, W - out_size)
    y = npr.randint(0, H - out_size)
    img = img[:, :, y:y+out_size, x:x+out_size]

    return img
