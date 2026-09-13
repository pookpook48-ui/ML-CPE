import os
import numpy as np
import json
from data_loader import load_image_paths
from preprocessing import preprocess_images
from split_data import split_dataset
from nn_model import build_and_train_nn
from evaluate import evaluate_model

os.makedirs("outputs", exist_ok=True)

image_paths, labels, classes = load_image_paths("PetImages")
features = preprocess_images(image_paths)
labels = np.array(labels)

np.save("outputs/features.npy", features)
np.save("outputs/labels.npy", labels)
with open("outputs/classes.json", "w") as f:
    json.dump(classes, f)

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(features, labels)

np.save("outputs/X_train.npy", X_train)
np.save("outputs/X_val.npy", X_val)
np.save("outputs/X_test.npy", X_test)
np.save("outputs/y_train.npy", y_train)
np.save("outputs/y_val.npy", y_val)
np.save("outputs/y_test.npy", y_test)

model, history = build_and_train_nn(X_train, y_train, X_val, y_val, X_train.shape[1])
evaluate_model(model, X_test, y_test)