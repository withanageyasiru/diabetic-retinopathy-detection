import json
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import Input
from tensorflow.keras.models import Model

from models.basemodel import BaseModel


class ModelHemorrhageAndHardExudates:

    config = None
    model = None
    model_name = 'model_hemorrhage_and_hard_exudates_10_14_2021'

    def __init__(self):
        self.create_model()
        # with open("models/model_hemorrhage_and_hard_exudates/hyperP.json", "r") as f:
        #     self.config = json.load(f)

    def create_model(self):
        print("Model is crating")
        # load model
        model = Sequential()

        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        out = model.layers[-1].output
        model = Model(input, out)
        model.summary(line_length=150)

        self.model = model
        # self.save_model_json()
        self.save_model_h5()

    def save_model_json(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open( self.model_name + ".json", "w") as json_file:
            json_file.write(model_json)

    def save_model_h5(self):
        # save model and architecture to single file
        self.model.save(self.model_name + ".h5")
        print("Saved model to disk")


model = ModelHemorrhageAndHardExudates()

