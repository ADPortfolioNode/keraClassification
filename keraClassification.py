import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

"""
This is the keraClassification module.
"""

# read the data
print('Loading the data...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('Data loaded...')
print(X_train.shape)

plt.imshow(X_train[0])

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2]  # find size of one-dimensional vector
print(num_pixels)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # flatten training images
print('training images flattened', X_train.shape)
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')  # flatten test images
print('test images flattened', X_test.shape)

# normalize inputs from 0-255 to 0-1
print('normalizing inputs...')
X_train = X_train / 255
X_test = X_test / 255

print('inputs normalized...')
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print(y_train[0])

num_classes = y_test.shape[1]
print('classes', num_classes)

# BUILD A NEURAL NETWORK
# define classification model

print('Building the model...')


def classification_model():
    """
    This function defines the classification model.
    """
    # create model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# TRAIN AND TEST THE NETWORK
# build the model
model = classification_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32, verbose=2)
print('Model trained...')
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('model scores:', scores)

# ACCURACY
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# save the model
keras.saving.save_model(model, 'mnist_model.keras')