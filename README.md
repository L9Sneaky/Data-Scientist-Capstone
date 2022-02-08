# Data Scientist Capstone
# Handwritten Greek Leters Prediction Model

### Dependencies
* Python 3+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn, TensorFlow, Keras
* Web App and Data Visualization: Flask, Plotly


## Project Definition
### Project Overview
For this project, we used Deep Learning to be able to predict greek letters, by using a pre-existing dataset from Kaggle (https://www.kaggle.com/sagyamthapa/handwritten-math-symbols).
### Problem Statement
Due to applications being hard to recognize greeks letters in mathematical equations, this project will be able to recognize greek letters in-between mathematical equations.
### Metrics
We use Training and Validation Accuracy and Loss to see our models preformance.

### Installing
Clone this GIT repository:
```
git clone https://github.com/L9Sneaky/Data-Scientist-Capstone.git
```

### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To clean the data
        `python models\data_preparation.py`
    - To run the ML that trains and saves the model
        `python models\model_init.py`

2. Run the following command in the app's directory to run your web app.
    `python app\app.py`

3. Go to http://0.0.0.0:3001/

## Analysis
### Data Exploration
Since the dataset used has alot of classes ive only picked 4 which are (Alpha, Beta, Pi, Theta) with a total of 9,700 observations, and since the dataset is composed of images, there is no need to remove any of them, and since the Dataset is balanced there is no need to tamper with the data.
### Data Visualization
![Data Visualization Pic](Screenshots/atom_RzH5HsqdP9.png)

## Results
### Model Evaluation and Validation
The dataset was split into training and testing set by 0.25 ratio and 0.2 validation ratio on the training set, the model used is a Convolution Neural network model with 3 convolution layers and 5 hidden layers and an output layer.
![Model Evaluation Pic](Screenshots/atom_hmGJf8fkBi.png)
### Validation
![Model Validation Pic](Screenshots/atom_4bNP52CcHK.png)

## Conclusion
In Conclusion the model can accurately predict 99% of the time but it is critical to have a bit of error to not be biased on prediction.
## Improvement
For future work, I would add more to the dataset to have more classes than 4 and have a better user-friendly UI to make it easier for the user to use.


## Authors

* [Ghanim Alghanim](https://github.com/L9Sneaky)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program

