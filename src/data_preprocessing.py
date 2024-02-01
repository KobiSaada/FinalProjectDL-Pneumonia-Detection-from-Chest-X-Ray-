import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set your dataset directory
dataset_dir = '../dataset'  # Make sure to update this path


def load_data(dataset_path, img_size=(224, 224)):
    """
    Load and preprocess the dataset.
    """
    datagen_train_val = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    # Load training data
    train_generator = datagen_train_val.flow_from_directory(
        os.path.join(dataset_path, 'train'),  # Corrected path
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='training'  # Set as training data
    )

    # Load validation data
    validation_generator = datagen_train_val.flow_from_directory(
        os.path.join(dataset_path, 'val'),  # Corrected path to use 'val' directory for validation
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
        # This line is adjusted to use the validation data correctly. If your 'val' directory isn't meant for splitting like training, remove the 'subset' argument
    )

    return train_generator, validation_generator


def preprocess_test_data(test_data_path, img_size=(224, 224)):
    """
    Preprocess the test data.
    """
    datagen_test = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen_test.flow_from_directory(
        test_data_path,
        target_size=img_size,
        batch_size=32,
        class_mode='binary'
    )

    return test_generator


if __name__ == "__main__":
    # Make sure these paths are correctly pointing to your dataset directories
    train_path = os.path.join(dataset_dir, 'train')
    val_path = os.path.join(dataset_dir, 'val')  # Ensure you have a validation set; if not, adjust accordingly
    test_path = os.path.join(dataset_dir, 'test')

    # Load and preprocess the training and validation data
    train_generator, validation_generator = load_data(dataset_dir)

    # Preprocess the test data
    test_generator = preprocess_test_data(test_path)

    print("Data preprocessing complete.")
