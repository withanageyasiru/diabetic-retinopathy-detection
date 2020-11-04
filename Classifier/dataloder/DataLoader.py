import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

class DataLoader:

    NUM_OF_IMAGES = 301
    CSV_PATH = "Classifier\data\Train\\train.csv"
    DEFAULT_DATA_PATH = "Classifier\data\Train\gaussian_filtered_images"

    def __init__(self):
        print("dataloader")

    def loadData(self):  
        train_df = [] 
        images = []
        
        data = pd.read_csv(self.CSV_PATH) 
        data = data.head(self.NUM_OF_IMAGES)
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        folder_path= os.path.join(fileDir, self.DEFAULT_DATA_PATH)
        file_names = data["id_code"].values
        for filename in file_names:
            filename = filename + ".png"           
            file_path= os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        x_train =  np. array(images , dtype=float)
        y_train =  np. array(data["diagnosis"].values)
        train_df = [x_train,y_train]
        return train_df



    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

    