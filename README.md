# Inertial Data Classification using SVM

This project processes raw inertial sensor data, extracts features, filters them, and classifies the resulting signals using a **Support Vector Machine (SVM)** model with **Leave-One-Out cross-validation**.

---

## 📂 Project Structure

├── extractfilesfinal.py # Parses raw .txt sensor data and extracts TSFEL features
├── Short_features.py # Filters relevant features and exports to CSV
├── SVM_Leave_one_out.py # Trains and evaluates the SVM model
├── Predict_new_samples.py # Applies the trained model to new unseen data
├── filtered_features.csv # Example filtered feature file
├── combined_features.csv # Full extracted feature dataset
└── data_raw/ # Folder containing the original .txt files

yaml
Copiar código

---

## ⚙️ Workflow Overview

### 1. Feature Extraction
`extractfilesfinal.py` parses the raw `.txt` sensor files, extracts **time-domain** features using the [TSFEL](https://tsfel.readthedocs.io/) library, and saves the combined output to `combined_features.csv`.

```python
from extractfilesfinal import extract
extract("path/to/data_raw")
2. Feature Filtering
Short_features.py filters the extracted dataset to retain only key features relevant for classification and saves them to filtered_features.csv.

python
Copiar código
from Short_features import filter_features
filter_features("path/to/data_raw")
3. Model Training & Evaluation
SVM_Leave_one_out.py trains a linear SVM using Leave-One-Out cross-validation and reports:

Model accuracy

Classification report

Confusion matrix

Run it with:

bash
Copiar código
python SVM_Leave_one_out.py
Example Results
Confusion Matrix:

Classification Report:

4. Prediction on New Data
Predict_new_samples.py applies the trained model to new unseen samples.
It uses the same extraction and filtering pipeline before generating predictions.

python
Copiar código
from Predict_new_samples import predict_new_samples
predict_new_samples("path/to/new_data")
Example Predictions

🧠 Dependencies
Install required packages with:

bash
Copiar código
pip install pandas scikit-learn tsfel matplotlib
🗂️ Data Description
Put your raw inertial measurement .txt files inside the data_raw/ folder.
Each file should include time-stamped accelerometer and gyroscope readings, for example:

makefile
Copiar código
Acc: ax, ay, az
Gyro: gx, gy, gz
t: timestamp
🧩 Typical Workflow Summary
Place your raw .txt files inside data_raw/

Run extractfilesfinal.py to extract features

Run Short_features.py to filter the dataset

Run SVM_Leave_one_out.py to train and evaluate the model

Use Predict_new_samples.py to classify new incoming data

✍️ Author
Diogo Sequeira
Experimental Physics / Data Analysis
