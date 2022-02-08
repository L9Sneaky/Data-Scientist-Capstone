import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

model = keras.models.load_model("model.model")
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
CATAGORIES = ["alpha","beta","pi","theta"]
#%%
IMG_SIZE = 64
bruh2 = cv2.imread('app/static/uploads/Untitled.jpg')
bruh2 = bruh2/255.0
bruh2 = cv2.resize(bruh2,(IMG_SIZE, IMG_SIZE))
bruh2 = np.expand_dims(bruh2, axis=0)


print(bruh2.shape)
#%%
predictions=model.predict(bruh2)
CATAGORIES[int(np.argmax(predictions))]
