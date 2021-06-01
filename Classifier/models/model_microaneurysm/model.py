import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import configparser


from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16


class ModelMicroaneurysm:
    config = configparser.ConfigParser()
    config.read('config.ini')

    config = None
    model = None
    model_name = 'model_microaneurysm_10_14_2021'

    def __init__(self):
        self.create_model()
        # with open("models/model_microaneurysm/hyperP.json", "r") as f:
        #     self.config = json.load(f)

    def create_model(self, dim= 32):
        print("Model is crating")
        # load model

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))




        # summarize the model
        model.summary()

        self.model = model
        self.save_model_json()
        self.save_model_h5()

    def save_model_json(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.model_name + ".json", "w") as json_file:
            json_file.write(model_json)

    def save_model_h5(self):
        # save model and architecture to single file
        self.model.save(self.model_name + ".h5")
        print("Saved model to disk")

model = ModelMicroaneurysm()