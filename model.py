import cv2
import joblib
import cv2, os
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



def openDataSet(route):
    paths = ['open','close']
    labels = []
    images = []
    target_shape = (128,128)
    for path in paths:
        pathToFile = route + '/' + path
        
        for filename in os.listdir(pathToFile):
            pil_image = Image.open(pathToFile + '/' + filename)
            image = np.array(pil_image)
            img = cv2.resize(image, target_shape)
            images.append(img)
            if path == 'open':
                labels.append(0)
            else:
                labels.append(1)
    images = np.array(images)
    images = np.array(images).reshape(-1, target_shape[0] * target_shape[1])
    
    return images, labels

def ApplyPCA(X):
    pca = PCA(n_components=6)
    pca_fit = pca.fit_transform(X)
    pca_components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_fit, pca_components


def model():
    route = "C:/Users/gabriele/Desktop/uni/exam/dataset"
    images, labels = openDataSet(route)
    images, components = ApplyPCA(images)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4)
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=50)  # Adjust 'C' and 'max_iter' as needed
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print('Accuracy:',accuracy)
    
    joblib.dump(model, 'C:/Users/gabriele/Desktop/uni/exam/model.pkl')
    joblib.dump(components,'C:/Users/gabriele/Desktop/uni/exam/pca_components.pkl')

model()