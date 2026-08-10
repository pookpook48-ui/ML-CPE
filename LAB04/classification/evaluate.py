from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd

def evaluate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def save_confusion_matrix(y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.savefig(os.path.join(output_dir, '02_confusion_matrix.png'))
    plt.close()

def save_k_curve(k_list, acc_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(k_list, acc_list, marker='o')
    plt.savefig(os.path.join(output_dir, '01_k_curve.png'))
    plt.close()

def save_predictions(X_test, y_test, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = X_test.copy()
    df['Actual'] = y_test.values
    df['Predicted'] = y_pred
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)