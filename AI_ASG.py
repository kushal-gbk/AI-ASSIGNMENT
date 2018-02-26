from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils

import struct
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl


np.random.seed(5)


(X_train, y_train), (X_test, y_test) = mnist.load_data()


'''
    Flatten the Numpy Array from 2D (28*28) to 1D (784*1)
'''
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')
'''
    Change Intensity Values of Pixels from 0 - 255 to 0 - 1
'''
X_train = X_train / 255
X_test = X_test / 255

'''
    One Hot Encoding . 
    i.e.,
        0 =  0000000001
        1 =  0000000010
        2 =  0000000100
        3 =  0000001000
        .
        .
        9 =  1000000000
'''
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


model = Sequential()

'''
    This is Input Layer
'''
model.add(Dense(784 , input_dim=784, kernel_initializer='random_uniform',activation='sigmoid'))

'''
    These are hidden layers . 16 Neurons in each Layer
'''
model.add(Dense(16 , activation='sigmoid'))
model.add(Dense(16 , activation='sigmoid'))

'''
    This is output Layer
'''
model.add(Dense(10 , activation='sigmoid'))


'''
    Compile the model
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


'''
    Fit the Model
'''
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

print("Final Baseline Error(Using Sigmoid): %.2f%%" % (100-scores[1]*100))

'''
    RELU 
'''

model_relu = Sequential()

model_relu.add(Dense(784 , input_dim=784, kernel_initializer='random_uniform',activation='relu'))

model_relu.add(Dense(16 , activation='relu'))
model_relu.add(Dense(16 , activation='relu'))

model_relu.add(Dense(10 , activation='sigmoid'))

model_relu.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_relu.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model_relu.evaluate(X_test, y_test, verbose=0)

print("Final Baseline Error(Using Relu): %.2f%%" % (100-scores[1]*100))