import pickle
import pandas as pd
import numpy as np

class LoadPickle:
    PIC_PATH = "Classifier\processed_data\data.pickle"
    def __init__(self):
        super().__init__()

    def loadPickle(self):
        with open(self.PIC_PATH, 'rb') as f:
            data = pickle.load(f)
        x_train =  np.array(data[0])

        y_train_encode = pd.get_dummies(data[1]).values
        y_train =  np.array(y_train_encode)
        return x_train,y_train