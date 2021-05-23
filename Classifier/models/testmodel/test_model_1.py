from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
import numpy as np

digit_a = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_a)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out_a = Flatten()(x)

digit_b = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_b)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out_b = Flatten()(x)

concatenated = concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)
model = Model([digit_a, digit_b], out)
print(model.summary())
model.compile('sgd', 'binary_crossentropy', ['accuracy'])
X = [np.zeros((1,27,27,1))] * 2
y = np.ones((1,1))
model.fit(X, y)