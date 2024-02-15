import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)

EPOCHS = 50
dataset_dir = '../dataset'  # Update with the correct path to your dataset

def preprocess_data(dataset_dir, subset_type):
    """
    Load and preprocess data from a given subset type ('train', 'val', 'test').
    """
    return image_dataset_from_directory(
        os.path.join(dataset_dir, subset_type),
        seed=123,
        shuffle=True,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'  # Assuming binary classification ('NORMAL', 'PNEUMONIA')
    )

# Loading datasets
train_dataset = preprocess_data(dataset_dir, 'train')
val_dataset = preprocess_data(dataset_dir, 'val')
test_dataset = preprocess_data(dataset_dir, 'test')

# Data Normalization (Scaling pixel values to [0, 1])
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data Augmentation (Optional, for training dataset)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Make sure to prefetch the datasets for optimal performance during training and testing
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
