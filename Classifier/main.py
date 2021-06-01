from dataloder.DataLoader import DataLoader
from dataloder.LoadPickle import LoadPickle
from preprocess.preprocess import Preprocess
from models.basemodel import BaseModel
from trainer.train import Train
from predict.predict import Predict
import configparser


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    get_raw_data = bool(config.get('MAIN', 'get_raw_data'))
    train = bool(config.get('MAIN', 'train'))

    if get_raw_data:
        train_data = DataLoader().load_data((512,512))
        preprocessor = Preprocess()
        preprocessor.preprocess(train_data)

    x_train, y_train = LoadPickle().load_pickle()

    if train:
        model = BaseModel()
        Train(x_train, y_train, model)

    # Predict()


if __name__ == "__main__":
    main()

