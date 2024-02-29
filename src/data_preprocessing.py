import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import resnet50


# Define constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
dataset_dir = '../dataset'  # Update with the correct path to your dataset

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# ResNet50 specific preprocessing
def preprocess_input_with_augmentation(x):
    x = data_augmentation(x, training=True)  # Apply data augmentation
    return resnet50.preprocess_input(x)  # Apply ResNet50 preprocessing

# Loading and preprocessing datasets
def preprocess_data(dataset_dir, subset_type):
    if subset_type == 'train':
        dataset = image_dataset_from_directory(
            os.path.join(dataset_dir, subset_type),
            seed=123,
            shuffle=True,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
            interpolation='bilinear',
        )
        # Apply augmentation only to the training dataset
        dataset = dataset.map(lambda x, y: (preprocess_input_with_augmentation(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # For validation and test sets, just apply ResNet50 preprocessing
        dataset = image_dataset_from_directory(
            os.path.join(dataset_dir, subset_type),
            seed=123,
            shuffle=True if subset_type == 'val' else False,  # Shuffle only validation set
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
            interpolation='bilinear',
        )
        dataset = dataset.map(lambda x, y: (resnet50.preprocess_input(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.cache().prefetch(tf.data.AUTOTUNE)

# Load datasets
train_dataset = preprocess_data(dataset_dir, 'train')
val_dataset = preprocess_data(dataset_dir, 'val')  # Assuming 'val' folder exists for validation data
test_dataset = preprocess_data(dataset_dir, 'test')  # Assuming 'test' folder exists for test data

