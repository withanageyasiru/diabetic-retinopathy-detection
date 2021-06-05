from dataloder.DataLoader_2 import DataLoader
from dataloder.LoadPickle import LoadPickle
from preprocess.preprocess import Preprocess
from models.basemodel import BaseModel
from trainer.train import Train
from predict.predict import Predict
import configparser
import numpy as np

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    get_raw_data = True #bool(config.get('MAIN', 'get_raw_data'))
    train = True #bool(config.get('MAIN', 'train'))
    x_train, y_train = None, None
    if get_raw_data:
        train_data = DataLoader().load_data((512,512))
        preprocessor = Preprocess()
        x_train, y_train = preprocessor.preprocess(train_data)



    if train:
        x_train, y_train = LoadPickle().load_pickle()
        x_train = [x_train[0] , x_train[1] / 255]
        x_train = [x_train[0].astype(np.float32), x_train[1].astype(np.float32)]
        x_test = x_train[1].astype(np.float32)[-100:]
        y_test = y_train[-100:]
        x_train = x_train[1].astype(np.float32)[:-100]
        y_train = y_train[:-100]

        model = BaseModel()
        Train(x_train, y_train, x_test , y_test, model)

    Predict()


if __name__ == "__main__":
    main()

