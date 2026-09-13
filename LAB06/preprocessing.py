import cv2
import numpy as np

def preprocess_images(image_paths, img_size=(64, 64)):
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        features.append(img)
    
    features = np.array(features, dtype='float32') / 255.0
    features = features.reshape(len(features), -1)
    return features