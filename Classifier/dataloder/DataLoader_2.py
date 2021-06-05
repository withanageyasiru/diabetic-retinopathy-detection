import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import configparser
from sklearn.utils import shuffle

from tqdm import tqdm


class DataLoader:
    config = configparser.ConfigParser()
    config.read('config.ini')

    NUM_OF_IMAGES = int(config.get('DATA_LOADER', 'NUM_OF_IMAGES'))
    CSV_PATH = config.get('DATA_LOADER', 'CSV_PATH')
    DEFAULT_DATA_PATH = config.get('DATA_LOADER', 'DEFAULT_DATA_PATH')

    def __init__(self):
        print("Data Loading...")

    def load_data(self, size):
        train_df = []
        images = []
        lables = []

        class_count = [0] * 5

        data = pd.read_csv(self.CSV_PATH)

        data = shuffle(data)
        data = data.head(self.NUM_OF_IMAGES)
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        folder_path = os.path.join(file_dir, self.DEFAULT_DATA_PATH)
        file_names = data["id_code"].values
        image_names = os.listdir(self.DEFAULT_DATA_PATH)
        for filename in tqdm(image_names):
             filename = filename.split('.')[0]
             if filename in file_names:
                index = np.where(data['id_code'] == filename)[0][0]
                level_ = data.iloc[index]['diagnosis']
                if class_count[level_] < 650:

                    tem_filename = filename
                    filename = filename + ".png"
                    file_path = os.path.join(folder_path, filename)
                    img = cv2.imread(file_path)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img,size)
                    images.append(img)
                    # index = data.index[data['image'] == filename].tolist()
                    # index = np.where(data['image'] == tem_filename )[0][0]
                    # level_ = data.iloc[index]['level']
                    lables.append(level_)
                    class_count[level_] += 1

        x_train = np.array(images, dtype=int)
        y_train = np.array(lables, dtype=int)
        train_df = [x_train, y_train]
        print("Data Loaded Successfully")
        return train_df

    def debug_display(self, image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

