import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D ,MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
#%%
print('Loading Data')
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X= X/255.0
print(X.shape[1:])

CATAGORIES = ["alpha","beta","gamma","lambda","phi","pi","sigma","theta"]
#%%
print('Initializing Model')
model = Sequential()


model.add(Conv2D(32, (3,3) , input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
#%%
n=15

model.fit(X, y, epochs=n ,batch_size=32,verbose=1, validation_split=0.2 , shuffle = True, use_multiprocessing = True)
model.save("model.model")
#%%
