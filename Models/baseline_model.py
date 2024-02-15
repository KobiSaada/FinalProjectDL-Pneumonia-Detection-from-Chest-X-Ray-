from src.data_preprocessing import preprocess_data  # Adjusted import statement

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
CHANNELS = 3
EPOCHS = 50
dataset_dir = '../dataset'  # Make sure this path is correct


# Define the logistic regression model
def build_logistic_regression_model():
    model = Sequential([
        Flatten(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Use the preprocess_data function to load datasets
train_dataset = preprocess_data(dataset_dir, 'train')
val_dataset = preprocess_data(dataset_dir, 'val')
test_dataset = preprocess_data(dataset_dir, 'test')

# Build the model
model = build_logistic_regression_model()

# Train the model
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
