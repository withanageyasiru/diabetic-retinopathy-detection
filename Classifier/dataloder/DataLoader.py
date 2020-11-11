import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import configparser


class DataLoader:
    config = configparser.ConfigParser()
    config.read('config.ini')

    NUM_OF_IMAGES = int(config.get('DATA_LOADER', 'NUM_OF_IMAGES'))
    CSV_PATH = config.get('DATA_LOADER', 'CSV_PATH')
    DEFAULT_DATA_PATH = config.get('DATA_LOADER', 'DEFAULT_DATA_PATH')

    def __init__(self):
        print("data_loader")

    def load_data(self):
        train_df = []
        images = []

        data = pd.read_csv(self.CSV_PATH)
        data = data.head(self.NUM_OF_IMAGES)
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        folder_path = os.path.join(file_dir, self.DEFAULT_DATA_PATH)
        file_names = data["id_code"].values
        for filename in file_names:
            filename = filename + ".png"
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        x_train = np.array(images, dtype=float)
        y_train = np.array(data["diagnosis"].values)
        train_df = [x_train, y_train]
        return train_df

    def debug_display(self, image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

