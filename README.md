# Data Scientist Capstone
# Handwritten Greek Leters Prediction Model

### Dependencies
* Python 3+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn, TensorFlow, Keras
* Web App and Data Visualization: Flask, Plotly


## Project Definition
### Project Overview
Converting paper to digital documents with the objective of better safeguarding old records is one of the issues we face today. As a result, programming machines to read ancient texts will improve efficiency. Using a pre-existing dataset from Kaggle, we applied Deep Learning to predict Greek letters (Referenced below).

### Problem Statement
We are given a dataset from Kaggle that has 10,000 grayscale images with the dimensions of 45 widths, 45 hights, and a depth of 1 since its greyscale. The model needs to be 0.9 accurate to be able to read greek letters in between mathematical equations.

### Metrics
We use Accuracy Score to know how accurate our model is. To accomplish that, we split the data into 3 sections, Training, Testing, and Validating.

We train the model using the training data, and test the accuracy using the testing data then we get the score using the equation below:
Accuracy Score = (Number of features classified correctly) / (Number of actual features)

The reason why we use accuracy score as our metric is that it's a convolution metric since our model is a classification prediction model.


## Analysis
### EDA
Since the dataset used has alot of classes ive only picked 4 which are (Alpha, Beta, Pi, Theta) with a total of 9,700 observations, and since the dataset is composed of images, there is no need to remove any of them, and since the Dataset is balanced there is no need to tamper with the data.
![Data Visualization Pic](Screenshots/atom_RzH5HsqdP9.png)


## Modeling
For preprocessing we didn't change much since the dimensions of the images are small. Furthermore, the colour of the data did not need to be reduced since its already greyscale and not RGB. So the next step was to split the data into training and testing set by 0.75 and 0.25 out of the data and for the validation it was 0.2 out of the training dataset. The model used is a Convolution Neural network model with 3 convolution layers and 5 hidden layers and an output layer. For the convolution layer, each layer extracts 32 features and 3x3 kernel and the pooling 2x2 max-pooling kernel. In addition, a ReLU activation function has been placed between each layer, except for the last layer (output) the function used is softmax considering that it’s a categorical output.

![Model Evaluation Pic](Screenshots/atom_hmGJf8fkBi.png)

![Model Validation Pic 1](Screenshots/atom_lCtq2YfUxI.png)

## Wep app
After running the app.py and opening the local website.

![Web app init pic 1](Screenshots/chrome_OIiidzIS3e.png)

upload the handwritten letter.

![Example pic 1](Screenshots/Untitled.jpg)

and result will show after pressing the upload button.

![Example pic 2](Screenshots/chrome_SYkheHt98V.png)


## Conclusion
In conclusion, we are trying to conversion of paper documents to digital documents with the goal of better safeguarding old records. Programming machine learning to interpret ancient writings will boost efficiency as a result. CNN was used to predict Greek letters using a pre-existing dataset from Kaggle. In the end, the model can successfully forecast 99 percent of the time, but a little margin of error is required to avoid relying on prediction.

## Improvement
For future work, I would add more to the dataset to have more classes than 4 and have a better user-friendly UI to make it easier for the user to use.


## Authors

* [Ghanim Alghanim](https://github.com/L9Sneaky)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program

## References
1. Dataset (https://www.kaggle.com/sagyamthapa/handwritten-math-symbols)
