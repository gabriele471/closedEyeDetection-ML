import joblib
import cv2, os
import numpy as np

from PIL import Image
from videoCapture import facialRecogniction


loaded_model = joblib.load('C:/Users/gabriele/Desktop/uni/exam/model.pkl')
loaded_pca_components = joblib.load('C:/Users/gabriele/Desktop/uni/exam/pca_components.pkl')

def main():
    facialRecogniction(loaded_model, loaded_pca_components)

if __name__ == "__main__":
    main()