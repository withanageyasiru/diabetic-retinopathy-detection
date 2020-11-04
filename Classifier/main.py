from dataloder.DataLoader import DataLoader
from dataloder.LoadPickle import LoadPickle
from preprocess.preprocess import Preprocess
from models.model_10_14_2020.model import Model
from trainer.train import Train
from predict.predict import Predict


def main():
    GET_RAW_DATA = True
    TRAIN = True

    if (GET_RAW_DATA):
        train_data = DataLoader().loadData()
        preprocessor = Preprocess()
        preprocessor.preprocess(train_data)

    x_train, y_train = LoadPickle().loadPickle()

    if (TRAIN):
        model = Model()
        train = Train(x_train, y_train, model)

    predict = Predict()


if __name__ == "__main__":
    main()
