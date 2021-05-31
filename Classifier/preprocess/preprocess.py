import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from preprocess.GLCM_features import createGLCMImage



class Preprocess:
    
    PIC_PATH = "processed_data/data.pickle"

    def __init__ (self):
        print("pre processing")



    def preprocess(self, train_data):
        # self.debug_display(x_train[5])
        # train_data[0] = train_data[0] / 225
        # train_data[1] = pd.get_dummies(train_data[1])

        image = train_data[0][:5]
        data = createGLCMImage(image)
        # print(data)

        #color intensity Component
        #noise removal
        #contrast Enhancement
        #vessel segmentation
        #optic disk segmentation



        return

    def save_pickle(self, data):
        pickle_out = open(self.PIC_PATH, "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def debugDisplay(self, image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()



