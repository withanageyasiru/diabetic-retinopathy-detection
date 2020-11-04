import cv2
import tensorflow as tf

class Predict:
    CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value

    def __init__(self):
        self.predict()

    def prepare(self,filepath):
        # e6f0ce5bf282,2
        # e724866f5084,2
        # e7291472109b,0
        # e740af6ac6ea,4
        # e756495c11cb,2
        # e7578d8dba72,0
        # e76a9cbb2a8c,3
        IMG_SIZE = 224  # 50 in txt-based

        img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        # return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


    def predict(self):
        model = tf.keras.models.load_model("build/model_10_14_2020")
        prediction = model.predict([self.prepare('data/Train/gaussian_filtered_images/e724866f5084.png')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
        print(prediction)