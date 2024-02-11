import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from src.data_preprocessing import load_dataset  # Adjust the import path as necessary

def build_logistic_regression_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),  # Flatten the input images to 1D vectors
        Dense(1, activation='sigmoid')  # Dense layer with 1 unit for binary classification
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    dataset_dir = '../dataset'  # Update this path to your dataset

    print("Loading and preprocessing data...")
    # Load training and validation data
    X_train, y_train = load_dataset(os.path.join(dataset_dir, 'train'))
    X_val, y_val = load_dataset(os.path.join(dataset_dir, 'val'))
    # Load test data
    X_test, y_test = load_dataset(os.path.join(dataset_dir, 'test'))

    print("Building Logistic Regression model...")
    model = build_logistic_regression_model(input_shape=(224, 224, 3))

    print("Training Logistic Regression model...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
