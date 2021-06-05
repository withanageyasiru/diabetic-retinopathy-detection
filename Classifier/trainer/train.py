import tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4524)])

    except:
        pass


class Train:

    learning_rate = None
    num_epochs = None
    model_name = None
    batch_size = None

    def __init__(self, x_train, y_train, x_test , y_test, model):
        super().__init__()        
        self.learning_rate = model.config["learning_rate"]
        self.num_epochs = model.config["num_epochs"]
        self.model_name = model.config["model_name"]
        self.batch_size = model.config["batch_size"]
        self.train(x_train, y_train, x_test , y_test, model.model)

    def train(self, x_train, y_train, x_test , y_test, model):
        # history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs, validation_split=0.3)
        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True, fill_mode="nearest", validation_split=0.2)
# ----------------------------------------------------------------------------------------------------------------------------
        my_training_batch_generator = TrainGenarator(x_train, y_train, self.batch_size)
        validation_generator = ValidationGenarator(x_test,y_test,self.batch_size)
        # train the network
        # history = model.fit_generator(aug.flow(x_train, y_train, batch_size=self.batch_size), steps_per_epoch=len(x_train[0]) // self.batch_size,
        #                         epochs=self.num_epochs)
        history = model.fit_generator(my_training_batch_generator,
                                      steps_per_epoch=len(x_train[0]) // self.batch_size,
                                      validation_data=validation_generator,
                                      validation_steps=len(x_test[0])// self.batch_size,
                                      epochs=self.num_epochs)

        # ----------------------------------------------------------------------------------------------------------------------
        model.save("build/" + self.model_name)
        self.plot_drafs(history, self.num_epochs)

    def plot_drafs(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


class TrainGenarator(tensorflow.keras.utils.Sequence):

    def __init__(self, train_x, train_y, batch_size):
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.train_x[0]) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # batch_x = [self.train_x[0][idx * self.batch_size: (idx + 1) * self.batch_size],self.train_x[1][idx * self.batch_size: (idx + 1) * self.batch_size]]
        batch_x = self.train_x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.train_y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_x, batch_y

class ValidationGenarator(tensorflow.keras.utils.Sequence):

    def __init__(self,  x_test , y_test, batch_size):
        self.test_x = x_test
        self.test_y = y_test
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.test_x[0]) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # batch_x = [self.train_x[0][idx * self.batch_size: (idx + 1) * self.batch_size],self.train_x[1][idx * self.batch_size: (idx + 1) * self.batch_size]]
        batch_x = self.test_x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.test_y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_x, batch_y


