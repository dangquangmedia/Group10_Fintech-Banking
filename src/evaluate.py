import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_confusion(y_true, y_pred, classes, out_path='reports/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def evaluate_model(model_path, X_test, y_test, classes=None):
    pipe = joblib.load(model_path)
    y_pred = pipe.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    if classes is None:
        classes = sorted(list(set(y_test)))
    plot_confusion(y_test, y_pred, classes)
    return rep
