# -*- coding: utf-8 -*-
"""
SVM Prediction on New Data Samples
----------------------------------
Loads new inertial data, extracts and filters features, scales them,
and performs predictions using the trained SVM model.

Created: Dec 19, 2024
Author: Diogo Sequeira
"""

from sklearn.preprocessing import StandardScaler
from extractfilesfinal import extract
from Short_features import filter_features
from SVM_Leave_one_out import modelo  # Trained SVM model from previous script


def predict_new_samples(samples_path: str) -> None:
    """
    Predicts the classes of new data samples using the trained SVM model.

    Parameters
    ----------
    samples_path : str
        Directory path containing the new `.txt` files to analyze.
    """
    # Extract and filter features from new data
    new_df = filter_features(samples_path)

    # Separate features
    X_new = new_df.drop(columns=['File', 'Class'])

    # Scale data (note: ideally use the same scaler fitted on training data)
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # Predict using trained model
    predictions = modelo.predict(X_new_scaled)

    # Display results
    print("\nPredictions for new samples:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Sample {i}: Predicted class = {pred}")


if __name__ == "__main__":
    # Example usage (replace with your path)
    samples_path = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\data"
    predict_new_samples(samples_path)
