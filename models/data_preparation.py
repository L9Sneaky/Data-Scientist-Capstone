import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

from rich.progress import track
#%%


inputos = "F:\Books\T10\sdaia t5\MTA Project\Data-Scientist-Capstone\data/"
DATADIR = inputos;
CATAGORIES = ["alpha","beta","gamma","lambda","phi","pi","sigma","theta"]

print(os.listdir(inputos))

#%%
IMG_SIZE = 45



for category in CATAGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))



training_data = []
def create_training_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()


random.shuffle(training_data)


#%%

X = []
y = []

for features, label in track(training_data):
    X.append(features)
    y.append(label)


X = np.array(X)/255.0

X = np.array(X)#.reshape(-1,IMG_SIZE,IMG_SIZE,1)

y = np.array(y)
#%%

#%%
pickle_out = open("X.pickle" , "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle" , "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
