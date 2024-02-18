import os
import tensorflow as tf

# Assuming 'preprocess_data' is properly defined in 'src.data_preprocessing'
from src.data_preprocessing import preprocess_data

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 13  # Might adjust based on observed performance
dataset_dir = '../dataset'  # Ensure this path is correctly specified


# Enhanced Data Preprocessing
def get_datasets():
    train_dataset = preprocess_data(dataset_dir, 'train')
    val_dataset = preprocess_data(dataset_dir, 'val')
    test_dataset = preprocess_data(dataset_dir, 'test')

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # Improved Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        # Additional augmentation layers can be added here
    ])

    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

    return (train_dataset.prefetch(tf.data.AUTOTUNE),
            val_dataset.prefetch(tf.data.AUTOTUNE),
            test_dataset.prefetch(tf.data.AUTOTUNE))


# Build an Enhanced CNN Model
def build_cnn_model():
    model = tf.keras.Sequential([
        # Input layer with explicit input shape
        tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        # Adding Batch Normalization to improve training stability
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),  # Adding Dropout to reduce overfitting
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Load and preprocess the data
train_dataset, val_dataset, test_dataset = get_datasets()

# Initialize and train the model
model = build_cnn_model()
model.summary()
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
