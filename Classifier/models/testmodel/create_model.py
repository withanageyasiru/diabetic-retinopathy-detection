
# import the necessary packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

class Models:
	def create_mlp(self, dim, regress=False):
		# define our MLP network
		model = Sequential()
		model.add(Dense(8, input_dim=dim, activation="relu"))
		model.add(Dense(4, activation="relu"))
		# check to see if the regression node should be added
		if regress:
			model.add(Dense(1, activation="linear"))
		# return our model
		return model


	def create_cnn(self, input_shape, filters=(16, 32, 64), regress=False):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering
		inputShape = input_shape
		chanDim = -1
		# define the model input
		inputs = Input(shape=inputShape)
		# loop over the number of filters

		x = inputs
			# CONV => RELU => BN => POOL
		x = Conv2D(33, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# flatten the volume, then FC => RELU => BN => DROPOUT
		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(0.5)(x)
		# apply another FC layer, this one to match the number of nodes
		# coming out of the MLP
		x = Dense(4)(x)
		x = Activation("relu")(x)
		# check to see if the regression node should be added

		model = Model(inputs, x)
		# return the CNN
		return model

# create the MLP and CNN models
models = Models()
mlp = models.create_mlp(4, regress=False)
cnn = models.create_cnn((640, 480, 3), regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)


# out = Dense(1, activation='sigmoid')(concatenated)
# model = Model([digit_a, digit_b], out)
print(model.summary())

model.compile('sgd', 'binary_crossentropy', ['accuracy'])
X = [np.ones((1,4)) , np.zeros((1,640, 480, 3)) ]
y = np.ones((1,1))

model.fit(X, y)
