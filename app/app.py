from distutils.log import debug
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

model = keras.models.load_model("F:\Books\T10\sdaia t5\MTA Project\Data-Scientist-Capstone\models\model.model")
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
CATAGORIES = ["alpha","beta","gamma","lambda","phi","pi","sigma","theta"]




UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))+'/static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            IMG_SIZE = 45
            bruh2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            bruh2 = bruh2/255.0
            bruh2 = cv2.resize(bruh2,(IMG_SIZE, IMG_SIZE))
            bruh2 = np.expand_dims(bruh2, axis=0)
            print(bruh2.shape)


            predictions=model.predict(bruh2)
            result=str(CATAGORIES[int(np.argmax(predictions))])
            return render_template('index.html', title='title', result = result)
            # return redirect(url_for('download_file', name=filename))
    return render_template('index.html', title='title')

if __name__ == '__main__':
    app.run(debug=True)
