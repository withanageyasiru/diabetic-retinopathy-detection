import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class Preprocess:
    def __init__ (self):
        print("preprocessing")

    def Preprocess(self,x_train,y_train):
        # self.debugDisplay(x_train[5])
        x_train = x_train / 255.0
        # x_train = x_train.reshape(x_train.shape[0],28,28,1)
        # print(x_train[1].mean())
        #Dump it(save it in binary format)
        # train_df = pd.DataFrame(dict({'image' : x_train, 'label' : y_train}))
        # with open('Classifier\processed_data\\data.pickle','wb') as fe_data_file:
        #     pickle.dump(train_df, fe_data_file)
        y_train = pd.get_dummies(y_train).values
        
        return x_train,y_train

    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

