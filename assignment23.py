import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ======================================
# Q1 - Used Bike Price Prediction
# Linear Regression
# ======================================

# Load Dataset
df = pd.read_csv("BIKE DETAILS.csv")

# Basic Information
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Remove Missing Values
df.dropna(inplace=True)

# Label Encoding
le = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

joblib.dump(encoders, "bike_encoders.pkl")

# Features and Target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Save Feature Names
joblib.dump(X.columns.tolist(), "columns.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error :", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# Save Model
joblib.dump(model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Linear Regression Model Saved Successfully")

# ======================================
# Q2 - IBM HR Employee Attrition
# Logistic Regression
# ======================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Basic Information
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Remove Missing Values
df.dropna(inplace=True)

# Label Encoding
le = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

joblib.dump(encoders, "hr_encoders.pkl")

# Features and Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Save Feature Names
joblib.dump(X.columns.tolist(), "hr_columns.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "logistic_scaler.pkl")

print("Logistic Regression Model Saved Successfully")

# ======================================
# Q3 - IBM HR Employee Attrition
# K-Nearest Neighbors (KNN)
# ======================================

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print("Accuracy :", accuracy_knn)
print("Precision :", precision_knn)
print("Recall :", recall_knn)
print("F1 Score :", f1_knn)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_knn))

print("\nClassification Report")
print(classification_report(y_test, y_pred_knn))

joblib.dump(knn_model, "knn_model.pkl")

print("KNN Model Saved Successfully")

# ======================================
# Q4 - IBM HR Employee Attrition
# Naive Bayes
# ======================================

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print("Accuracy :", accuracy_nb)
print("Precision :", precision_nb)
print("Recall :", recall_nb)
print("F1 Score :", f1_nb)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_nb))

print("\nClassification Report")
print(classification_report(y_test, y_pred_nb))

joblib.dump(nb_model, "naive_bayes_model.pkl")

print("Naive Bayes Model Saved Successfully")

# ======================================
# Q5 - Comparison of Models
# Logistic Regression vs KNN vs Naive Bayes
# ======================================

comparison = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Naive Bayes"
    ],
    "Accuracy": [
        accuracy,
        accuracy_knn,
        accuracy_nb
    ],
    "Precision": [
        precision,
        precision_knn,
        precision_nb
    ],
    "Recall": [
        recall,
        recall_knn,
        recall_nb
    ],
    "F1 Score": [
        f1,
        f1_knn,
        f1_nb
    ]
})

print("\nComparison of Models")
print(comparison)