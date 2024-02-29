import tensorflow as tf
from src.data_preprocessing import preprocess_data  # Assuming your preprocessing code is saved here
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import l2
# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
CHANNELS = 3
EPOCHS = 10  # Adjusted for demonstration, consider more epochs based on performance
dataset_dir = '../dataset'  # Ensure this is the correct path to your dataset

# Define the logistic regression model
def build_logistic_regression_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), regularization_factor=0.01):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(regularization_factor)),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the datasets
train_dataset = preprocess_data(dataset_dir, 'train')
val_dataset = preprocess_data(dataset_dir, 'val')
test_dataset = preprocess_data(dataset_dir, 'test')

# Build the model
model = build_logistic_regression_model()

# Train the model
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
model.summary()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
