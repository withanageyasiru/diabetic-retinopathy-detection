import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess.GLCM_features import createGLCMImage
from preprocess.color_intensity_component import intensity_component
from preprocess.contrast_enhansement import contrast_enhancement
from preprocess.noise_removal import noise_removal


class Preprocess:
    PIC_PATH = "processed_data/data.pickle"

    def __init__(self):
        print("pre processing ")

    def preprocess(self, train_data):
        # self.debug_display(x_train[5])
        # train_data[0] = train_data[0] / 225
        # train_data[1] = pd.get_dummies(train_data[1])

        image = np.uint8( train_data[0] )
        GLCM_data = [255,128]#createGLCMImage(image)
        # print(data)
        # self.debugDisplay(GLCM_data[0][:,:,0])
        # self.debugDisplay(GLCM_data[0][:, :, 1])
        # self.debugDisplay(GLCM_data[0][:, :, 2])

        # color intensity Component
        # image = intensity_component(image)
        # # self.debugDisplay(image[0])
        # # noise removal
        # image = noise_removal(image)
        # # self.debugDisplay(image[0])
        # # contrast Enhancement
        # image = contrast_enhancement(image)
        # self.debugDisplay(image[0])
        # vessel segmentation
        # vessel_seg_images = vessel_segmentation(image)

        # cv2.imshow("show", image[1, :, :, :])
        # cv2.waitKey(0)
        # #optic disk segmentation
        #
        # cv2.imshow("show", image[1, :, :, :])
        # cv2.waitKey(0)
        data = [[GLCM_data, image],train_data[1]]
        print("Preprocessing completed")
        self.save_pickle(data)
        return data

    def save_pickle(self, data):
        print("Data saving to pickel...")
        try:
            pickle_out = open(self.PIC_PATH, "wb")
            pickle.dump(data, pickle_out)
            pickle_out.close()
            print("Data save completed")
        except:
            print("Data save Not completed")

    def debugDisplay(self, image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()



