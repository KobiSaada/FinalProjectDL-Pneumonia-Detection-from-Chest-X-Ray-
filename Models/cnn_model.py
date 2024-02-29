import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# If 'preprocess_data' is a custom function you wrote, make sure it's correctly implemented
# For this example, we'll assume it returns preprocessed and batched datasets
from src.data_preprocessing import preprocess_data

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 13  # Adjust based on observed performance
dataset_dir = '../dataset'  # Ensure this path is correctly specified

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

def build_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tuning more layers
    base_model.trainable = True
    fine_tune_at = 100  # Fine-tune from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=True)  # Enable training=True for fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def calculate_f1_score(y_true, y_pred):
    # Generate a classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    # Extract the F1 score for the positive class
    f1 = report['1']['f1-score']
    return f1


# Display a batch of predictions
def display_predictions(dataset, model):
    images, labels = next(iter(dataset))  # Get one batch of images and labels
    predictions = model.predict(images)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        actual = labels[i].numpy()[0]
        predicted = predictions[i][0]
        plt.title(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")
        plt.axis("off")
    plt.show()

# Assuming 'preprocess_data' correctly prepares the datasets
train_dataset, val_dataset, test_dataset = preprocess_data(dataset_dir, 'train'), preprocess_data(dataset_dir, 'val'), preprocess_data(dataset_dir, 'test')

model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
compile_model(model)



callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.2f}")

# Display predictions for the test dataset
display_predictions(test_dataset, model)
# After model evaluation
# Assuming binary classification and test_dataset is properly batched
y_true = [labels.numpy() for _, labels in test_dataset]
y_pred = [model.predict(images) for images, _ in test_dataset]

# Flatten the lists
y_true = np.concatenate(y_true).astype(int)
y_pred = np.round(np.concatenate(y_pred)).astype(int)

# Display Confusion Matrix and calculate F1 Score
plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Pneumonia'], normalize=True)
f1_score = calculate_f1_score(y_true, y_pred)
print(f"F1 Score: {f1_score:.2f}")
plt.show()