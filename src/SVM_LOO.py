# -*- coding: utf-8 -*-
"""
Support Vector Machine (SVM) Classification Script
--------------------------------------------------
Performs classification using a linear SVM model with Leave-One-Out cross-validation
on pre-filtered inertial sensor features. Saves the final trained model for reuse.

Created: Dec 18, 2024
Author: Diogo Sequeira
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def run_svm_classification(file_path: str = 'filtered_features.csv') -> None:
    """
    Trains and evaluates an SVM classifier using Leave-One-Out cross-validation,
    then saves the trained model as 'svm_model.pkl' in the same directory.

    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file containing filtered features. Default is 'filtered_features.csv'.
    """
    # Load dataset
    df = pd.read_csv(file_path, sep=';')

    # Separate features (X) and labels (y)
    X = df.drop(columns=['File', 'Class'])
    y = df['Class']

    # Initialize linear SVM
    svm_model = SVC(kernel='linear')

    # Leave-One-Out cross-validation
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        svm_model.fit(X_train, y_train)
        prediction = svm_model.predict(X_test)[0]

        y_pred.append(prediction)
        y_true.append(y_test.iloc[0])

    # Final model training (for export)
    svm_model.fit(X, y)

    # Evaluation
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=svm_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Save the trained model
    model_path = Path(file_path).parent / "svm_model.pkl"
    joblib.dump(svm_model, model_path)
    print(f"Trained model saved successfully at: {model_path}")


if __name__ == "__main__":
    run_svm_classification()
