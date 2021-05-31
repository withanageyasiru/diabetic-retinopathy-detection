import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from preprocess.color_intensity_component import intensity_component

from preprocess.GLCM_features import createGLCMImage
from preprocess.contrast_enhansement import contrast_enhancement
from preprocess.noise_removal import noise_removal
from preprocess.vessel_segmentation import vessel_segmentation


class Preprocess:
    
    PIC_PATH = "processed_data/data.pickle"

    def __init__ (self):
        print("pre processing")



    def preprocess(self, train_data):
        # self.debug_display(x_train[5])
        # train_data[0] = train_data[0] / 225
        # train_data[1] = pd.get_dummies(train_data[1])

        image = train_data[0][:5]
        # data = createGLCMImage(image)
        # print(data)

        #color intensity Component
        image = intensity_component(image)
        #noise removal
        image = noise_removal(image)
        #contrast Enhancement
        image = contrast_enhancement(image)
        #vessel segmentation
        vessel_seg_images = vessel_segmentation(image)

        cv2.imshow("show", vessel_seg_images[1, :, :, :])
        cv2.waitKey(0)
        #optic disk segmentation

        cv2.imshow("show", image[1, :, :, :])
        cv2.waitKey(0)



        return

    def save_pickle(self, data):
        pickle_out = open(self.PIC_PATH, "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def debugDisplay(self, image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()



