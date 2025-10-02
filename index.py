# Task 7: Support Vector Machines (SVM)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
def load_dataset(path, target_column):
    data = pd.read_csv(path)
    if data.isnull().sum().any():
        data = data.dropna()
    X = data.drop(columns=[target_column, "id"])
    y = data[target_column]
    return X, y

# Scale features
def preprocess_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Train and evaluate SVM
def train_svm(X_train, X_test, y_train, y_test, kernel="linear", C=1.0, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM ({kernel} kernel) Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model, acc

# Cross validation
def cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"\nCross-validation scores: {scores}")
    print(f"Average CV Accuracy: {scores.mean():.4f}")

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 0.01, 0.001]}
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    print("\nBest Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)
    return grid.best_estimator_

# Predict random sample
def random_prediction(model, X, scaler):
    random_sample = {}
    for col in range(X.shape[1]):
        min_val, max_val = X.iloc[:, col].min(), X.iloc[:, col].max()
        random_sample[X.columns[col]] = np.random.uniform(min_val, max_val)
    sample_df = pd.DataFrame([random_sample])
    sample_scaled = scaler.transform(sample_df)
    pred = model.predict(sample_scaled)[0]
    print("\nRandom Sample Prediction")
    print("Sample values:\n", sample_df.to_dict(orient="records")[0])
    print("Prediction:", "Malignant (M)" if pred == 'M' else "Benign (B)")

# Main run
def main():
    X, y = load_dataset("breast-cancer.csv", "diagnosis")
    X_scaled, scaler = preprocess_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    linear_model, _ = train_svm(X_train, X_test, y_train, y_test, kernel="linear")
    rbf_model, _ = train_svm(X_train, X_test, y_train, y_test, kernel="rbf")

    cross_validation(linear_model, X_scaled, y)
    cross_validation(rbf_model, X_scaled, y)

    best_rbf = tune_hyperparameters(X_train, y_train)

    random_prediction(rbf_model, X, scaler)

if __name__ == "__main__":
    main()
