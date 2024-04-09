import os
from tkinter import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Rest of  code

# read the data
print('Loading the data...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('Data loaded...')
print(X_train.shape)

plt.imshow(X_train[0])

# flatten images into one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2]
# find size of one-dimensional vector
print(num_pixels)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# flatten training images
print('training images flattened', X_train.shape)
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# flatten test images
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


def classification_model(model):
    """
    This function defines the classification model.
    """
    # Adjust this to match the shape of your input data
    model.add(Input(shape=(784,)))
    model.add(Dense(num_classes, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # show model summary and return the model
    model.summary()
    return model


# TRAIN AND TEST THE NETWORK
# build the model
model = Sequential()
model = classification_model(model)
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print('weights:', weights)
print('bias:', bias)

model.summary()

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32, verbose=2)

print('Model trained...')
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=2, batch_size=200, steps=100)

print('model scores:', scores)

# ACCURACY
print(f'Accuracy: {scores[1]}% \n Error: {1 - scores[1]}')

# make predictions
print('Making predictions...')
predictions = model.predict(X_test)
plt.imshow(X_test[0].reshape(28, 28))
plt.xlabel('Actual:' + str(y_test[0]) + '\n' + 'Predicted:' + str(predictions[0]))


model.report = 'Model trained with an accuracy of {}%'.format(scores[1]), 'Error: {}'.format(1 - scores[1]), 'Predictions:', predictions, 'Model Summary:', model.summary(), 'Model Weights:', weights, 'Model Bias:', bias, 'Model Scores:', scores, 'Model Evaluation:', model.evaluate(X_test, y_test, verbose=2, batch_size=200, steps=100), 'Model Predictions:', model.predict(X_test), 'Model Image:', plt.imshow(X_test[0].reshape(28, 28)), 'Model Image Label:', plt.xlabel('Actual:' + str(y_test[0]) + '\n' + 'Predicted:' + str(predictions[0])), 'Model Summary:', model.summary()

# plot the results
plt.show()


# print the model report

print(f'Model Report : \n Model trained with an accuracy of {scores[1]}%', model.report)

# save the model
print('Saving the model...')
#
keras.models.save_model(model, 'mnist_model.keras')