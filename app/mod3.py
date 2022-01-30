import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

model = keras.models.load_model("F:/Books/T10/sdaia t5/MTA Project/Data-Scientist-Capstone/models/model.model")
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
CATAGORIES = ["alpha","beta","gamma","lambda","phi","pi","sigma","theta"]

os.system('cls')
while(True):
    IMG_SIZE = 45
    bruh2 = cv2.imread('F:/Books/T10/sdaia t5/MTA Project/Data-Scientist-Capstone/app/static/uploads/Untitled.jpg')
    bruh2 = bruh2/255.0
    bruh2 = cv2.resize(bruh2,(IMG_SIZE, IMG_SIZE))
    bruh2 = np.expand_dims(bruh2, axis=0)
    print(bruh2.shape)

    #os.system('cls')
    predictions=model.predict(bruh2)
    print(predictions)
    print(CATAGORIES[int(np.argmax(predictions))])
    time.sleep(3)

#plt.title(CATAGORIES[int(np.argmax(predictions))])
#plt.imshow(cv2.imread('Untitled.jpg'))
#plt.show()
