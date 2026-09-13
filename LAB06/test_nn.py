import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json

model = load_model("outputs/nn_model.keras")
X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

with open("outputs/classes.json", "r") as f:
    classes = json.load(f)
    
inv_classes = {v: k for k, v in classes.items()}

idx = np.random.choice(len(X_test), 4, replace=False)
X_sample = X_test[idx]
y_true = y_test[idx]
y_pred = (model.predict(X_sample) > 0.5).astype("int32").flatten()

fig, axes = plt.subplots(1, 4, figsize=(15, 3))
for i, ax in enumerate(axes):
    img = X_sample[i].reshape(64, 64, 3)
    ax.imshow(img)
    ax.set_title(f"T: {inv_classes[y_true[i]]} | P: {inv_classes[y_pred[i]]}")
    ax.axis('off')
plt.savefig("outputs/prediction_sample.png")
plt.show()