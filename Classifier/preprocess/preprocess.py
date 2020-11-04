import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class Preprocess:
    
    PIC_PATH = "processed_data/data.pickle"

    def __init__ (self):
        print("preprocessing")

    def preprocess(self,train_data):
        # self.debugDisplay(x_train[5])
        train_data[0] = train_data[0] / 225
        train_data[1] =  pd.get_dummies(train_data[1])

        pickle_out = open(self.PIC_PATH,"wb")
        pickle.dump(train_data, pickle_out)
        pickle_out.close()
        
        return 

    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

