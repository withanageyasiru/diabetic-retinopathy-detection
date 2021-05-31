import cv2
from util.fast_glcm import fast_glcm_std, fast_glcm_max, fast_glcm_entropy
import numpy as np


def createGLCMImage(images):

    glcm_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (300,300))
        h,w = img.shape


        std = fast_glcm_std(img)
        ma = fast_glcm_max(img)
        ent = fast_glcm_entropy(img)

        needed_multi_channel_img = np.zeros((img.shape[0], img.shape[1], 3))

        needed_multi_channel_img[:, :, 0] = std
        needed_multi_channel_img[:, :, 1] = ma
        needed_multi_channel_img[:, :, 2] = ent

        glcm_images.append(needed_multi_channel_img.astype(int))

    return np.array(glcm_images)
