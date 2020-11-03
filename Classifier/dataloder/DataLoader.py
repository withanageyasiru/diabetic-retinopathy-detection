import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle

class DataLoader:
    def __init__(self):
        print("dataloader")

    def loadData(self,path = "data\Train"):
        images = []
        import pandas as pd 
        data = pd.read_csv("Classifier\data\Train\\train.csv") 
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        folder_path= os.path.join(fileDir, path)
        print(data["id_code"].values)
        file_names = data["id_code"].values
        for filename in file_names:
            filename = filename + ".png"           
            file_path= os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append( img )
        x_train =  np. array(images , dtype=float)
        y_train =  np. array(data["diagnosis"].values)
        print(x_train[:300,:].shape)
        return x_train[:300,:],y_train[:300]

    # def loadDataOld(self,path = "data\Train"):
    #     fileDir = os.path.dirname(os.path.realpath('__file__'))
    #     folder_path= os.path.join(fileDir, path)
    #     folder_names=os.listdir(folder_path)
    #     for folder in folder_names:
    #         data_path= os.path.join(folder_path, folder)
    #         folder_names=os.listdir(data_path)
    #         labels=[(0 if re.findall(r"[\w']+", i)[0]=="circles"  else 1) for i in filenames]
    #         train_df = pd.DataFrame(dict({'filename' : filenames, 'class' : labels}))

    #         dataSet = train_df.sample(frac=1) # Shuffle data
    #         file_path = [os.path.join( data_path , i) for i in dataSet.filename.tolist()] 
    #         images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in file_path]
    #     x_train =  np.array(images)
    #     y_train =  dataSet['class'].to_numpy()
    #     return x_train,y_train 


    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

    def loadPickle(self):
        with open('Classifier\data\Train\gaussian_filtered_images\export.pkl', 'rb') as f:
            data = pickle.load(f)
        print (data)
        return data