import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
dataset_dir = '../dataset'  # Update with the correct path to your dataset

def preprocess_data(dataset_dir, subset_type):
    """
    Load and preprocess data from a given subset type ('train', 'val', 'test').
    """
    dataset = image_dataset_from_directory(
        os.path.join(dataset_dir, subset_type),
        seed=123,
        shuffle=True,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'  # Assuming binary classification ('NORMAL', 'PNEUMONIA')
    )
    return dataset.cache()  # Cache the images after loading

# Enhanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Custom Preprocessing Function
def custom_preprocess(image, label):
    image = data_augmentation(image, training=True)  # Apply data augmentation
    image = tf.keras.layers.Rescaling(1./255)(image)  # Normalize images
    return image, label

# Loading and preprocessing datasets
train_dataset = preprocess_data(dataset_dir, 'train').map(custom_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = preprocess_data(dataset_dir, 'val').map(custom_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = preprocess_data(dataset_dir, 'test').map(custom_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetching datasets
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

