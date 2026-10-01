import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train', color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='validation', color='#ff7f0e')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train', color='#1f77b4')
    plt.plot(history.history['val_loss'], label='validation', color='#ff7f0e')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def plot_prediction_samples(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    plt.suptitle("Prediction: Sample Results", fontsize=16)
    
    random_indices = np.random.choice(X_test.shape[0], 4, replace=False)
    
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 2, i + 1)
        
        img = X_test[idx]
        true_label = class_names[np.argmax(y_test[idx])]
        pred_label = class_names[np.argmax(y_pred[idx])]
        pred_prob = np.max(y_pred[idx]) * 100
        
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Pred: {pred_label} ({pred_prob:.0f}%)\nTrue: {true_label}", color=color)
        
    plt.tight_layout()
    plt.show()