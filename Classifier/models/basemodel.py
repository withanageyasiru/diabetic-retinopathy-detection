import json

from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import model_from_json, load_model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam



class BaseModel():


    def __init__(self):
        with open("models/hyperP.json", "r") as f:
            self.config = json.load(f)

        self.model = None
        self.__model_microaneurysm = self.load_model_from_json('models/model_microaneurysm/model_microaneurysm_10_14_2021.json')
        self.__model_hemorrhage_and_hard_exudates = self.load_model_from_h5('models/model_hemorrhage_and_hard_exudates/model_hemorrhage_and_hard_exudates_10_14_2021.h5')
        # self.__model_combine = self.load_model_from_json('models/model_combine/model_combine_10_14_2021.json')
        # self.__model_numerical_features = self.load_model_from_json('models/model_hemorrhage_and_hard_exudates/model_microaneurysm.json')
        self.base_model()

    def base_model(self):
        '''
        this function combine the
        :return:
        '''

        # create the input to our final set of layers as the *output* of both
        # the MLP and CNN
        combinedInput = concatenate([self.__model_microaneurysm.output, self.__model_hemorrhage_and_hard_exudates.output])
        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        out = Flatten()(combinedInput)
        x = Dense(64, activation="relu")(out)
        x = Dense(16, activation="relu")(x)
        x = Dense(5, activation="softmax")(x)
        # our final model will accept categorical/numerical data on the MLP
        # input and images on the CNN input, outputting a single value (the
        # predicted price of the house)
        self.model = Model(inputs=[self.__model_microaneurysm.input, self.__model_hemorrhage_and_hard_exudates.input], outputs=x)

        opt = Adam(lr=0.001)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        # out = Dense(1, activation='sigmoid')(concatenated)
        # model = Model([digit_a, digit_b], out)
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        self.model.summary()


    def load_model_from_json(self, json_file_name):
        # load json and create model
        json_file = open(json_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.summary()
        return loaded_model

    def load_model_from_h5(self, model_name):
        model = load_model(model_name, compile=False)
        model.summary()
        return model



