import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Assuming load_dataset and flatten_images are defined in your data_preprocessing script
from src.data_preprocessing import load_dataset


def build_logistic_regression_model(input_shape):
    """
    Build a logistic regression model using TensorFlow/Keras.
    """
    model = Sequential([
        Flatten(input_shape=input_shape),  # Flatten the input images to 1D vectors
        Dense(1, activation='sigmoid')  # Dense layer with 1 unit for binary classification
    ])

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    dataset_dir = '../dataset'  # Update this path to your dataset

    print("Loading and preprocessing data...")
    X_train, y_train = load_dataset(os.path.join(dataset_dir, 'train'))  # Load training data
    X_val, y_val = load_dataset(os.path.join(dataset_dir, 'val'))  # Load validation data

    # Assuming your images are 224x224 and RGB (3 channels)
    model = build_logistic_regression_model(input_shape=(224, 224, 3))

    print("Training Logistic Regression model...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Evaluate the model
    _, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
