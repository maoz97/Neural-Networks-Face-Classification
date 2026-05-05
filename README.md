# Neural-Networks-Face-Classification

The project is divided into two main parts:

# 1. Multi-Layer Perceptron (MLP) Experiments
Files: MLP.py, helpers.py

In this part, I built a Multi-Layer Perceptron using PyTorch to classify geographical data (European countries based on longitude and latitude).

Features custom dataset handling (EuropeDataset).

Includes various experiments to test how different hyperparameters affect the model: learning rate, network depth and width, batch sizes, and number of epochs.

Explores the effect of adding Batch Normalization and implicit representation.

Uses helpers.py to plot and visualize the decision boundaries of the different models.

# 2. Real vs. Fake Face Detection (CNN & XGBoost)
Files: cnn.ipynb, xg.py

This section focuses on classifying images from the "Which Face is Real?" dataset.

Trains a baseline XGBoost classifier on flattened image features.

Uses a pre-trained ResNet18 model to extract features and run Linear Probing.

Fine-tunes the ResNet18 model to improve accuracy.

Visualizes the test results to see examples of where the model guesses right and wrong.

# How to Run
To run the code, you will need to add the following dataset files to the main directory (these are not included in the repository):

train.csv, validation.csv, test.csv (for the MLP part).

The whichfaceisreal dataset (for the CNN part).

# You will also need to install the required libraries:
pip install torch torchvision numpy pandas matplotlib scikit-learn xgboost tqdm

# Credit and Copyright
This project was written as part of the Introduction to Machine Learning (IML) course. All rights to the assignment instructions and original course materials are reserved to the Hebrew University of Jerusalem.
