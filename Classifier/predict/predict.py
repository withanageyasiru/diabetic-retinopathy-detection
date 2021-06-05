import cv2
import tensorflow as tf
import configparser


class Predict:
    config = configparser.ConfigParser()
    config.read('config.ini')
    CATEGORIES = ["Dog", "Cat"]

    def __init__(self):
        self.predict()

    def prepare(self, filepath):
        # e6f0ce5bf282,2
        # e724866f5084,2
        # e7291472109b,0
        # e740af6ac6ea,4
        # e756495c11cb,2
        # e7578d8dba72,0
        # e76a9cbb2a8c,3
        img_size = 512  # 50 in txt-based

        img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
        # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array,(img_size, img_size))
        return img_array.reshape(-1, img_size, img_size, 3)
        # new_array = cv2.resize(img_array, (img_size, img_size))  # resize image to match model's expected sizing
        # return new_array.reshape(-1, img_size, img_size, 1)  # return the image with shaping that TF wants.

    def predict(self):
        model_file_path = self.config.get('PREDICT', 'model_file_path')
        model = tf.keras.models.load_model(model_file_path)

        # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
        image_path = self.config.get('PREDICT', 'image_path')
        prediction = model.predict([self.prepare(image_path)])
        print(prediction)

