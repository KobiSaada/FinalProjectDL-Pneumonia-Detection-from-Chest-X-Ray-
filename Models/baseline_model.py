import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure the directory of the data_preprocessing script is in the path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join('..', 'src'))

from data_preprocessing import load_dataset, flatten_images  # Import the preprocessing functions

dataset_dir = '../dataset'  # Update this path to your dataset

def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate the baseline logistic regression model.
    """
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline Model Accuracy: {accuracy * 100:.2f}%")

    return model

if __name__ == '__main__':
    print("Loading and preprocessing data...")
    # Load and preprocess the data
    X, y = load_dataset(os.path.join(dataset_dir, 'train'))  # Adjust path as necessary
    X = flatten_images(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the baseline model
    model = train_baseline_model(X_train, y_train, X_test, y_test)
