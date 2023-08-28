import joblib
import cv2, os
import numpy as np

from PIL import Image


def prepareImage(image, loaded_pca_components):
    target_shape = (128,128)
    image = np.array(image)
    img = cv2.resize(image, target_shape)
    #gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    images = np.array(img)
    images = np.array(images).reshape(-1, target_shape[0] * target_shape[1])
    pca_transformed_matrix = np.dot(images, loaded_pca_components.T)
    
    return pca_transformed_matrix

def makePrediction(matrix, loaded_model):
    y_pred = loaded_model.predict(matrix)
    predicted_probabilities = loaded_model.predict_proba(matrix)  # Predicted probabilities for each class
    
    return y_pred
    