# Brain Tumor Classification
This project aims to predict whether a person has glioma, meningioma, or no tumor using a Convolutional Neural Network (CNN). The project involves running the model in a Jupyter notebook, saving it, and then loading the saved model into a backend script to serve predictions through a web interface.

# Table of Contents
Overview
> Dataset
> Preprocessing
> Modeling
> Evaluation
> Saving and Loading the Model
> Web Application

# How to Run
# Requirements
Overview
Brain tumor classification is a crucial task in medical imaging to assist in diagnosis and treatment planning. This project utilizes a CNN to classify brain MRI images into three categories: glioma, meningioma, or no tumor.

# Dataset
The dataset used for this project consists of MRI images labeled as glioma, meningioma, or no tumor. Each image has been preprocessed and augmented to improve the model's performance.

# Preprocessing
Several preprocessing steps have been applied to the dataset:
Resizing: All images are resized to a fixed dimension suitable for the CNN model.
Normalization: Pixel values are normalized to the range [0, 1].
Augmentation: Techniques like rotation, zoom, and horizontal flip are applied to augment the dataset.

# Modeling
A Convolutional Neural Network (CNN) is used for this classification task. The model architecture includes several convolutional layers followed by max-pooling layers and fully connected layers. The final layer uses a softmax activation function to output probabilities for each class.

# How to Run
Run the Jupyter Notebook: Train the model and save it.
Run the Flask App: Load the saved model and serve the web application.

# Requirements
Python 3.x
Flask
TensorFlow
Keras
NumPy
Pandas
Matplotlib
