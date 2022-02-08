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
CATAGORIES = ["alpha","beta","pi","theta"]
