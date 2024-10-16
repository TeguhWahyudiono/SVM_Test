# Import library yang dibutuhkan
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load dataset Iris
data = load_iris()
X = data.data  # fitur
y = data.target  # label

# Pisahkan data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Buat dan latih model SVM
svm_model = SVC(kernel='linear')  # Kernel bisa diubah ke 'rbf', 'poly', dll.
svm_model.fit(X_train, y_train)

# Prediksi dengan model SVM
y_pred = svm_model.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
