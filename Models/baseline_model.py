import os
import tensorflow as tf
from src.data_preprocessing import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
CHANNELS = 3
EPOCHS = 20
dataset_dir = '../dataset'

# Logistic regression model with dropout and data augmentation
def build_logistic_regression_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), regularization_factor=0.001):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(0.5))  # Keep dropout as 0.5
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(regularization_factor)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(regularization_factor)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Load the datasets with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    dataset_dir + '/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_dataset = val_datagen.flow_from_directory(
    dataset_dir + '/val',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    dataset_dir + '/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Build the model
model = build_logistic_regression_model()

# Train the model
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()