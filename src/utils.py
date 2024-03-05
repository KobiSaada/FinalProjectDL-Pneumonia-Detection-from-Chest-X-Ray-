import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

# If 'preprocess_data' is a custom function you wrote, make sure it's correctly implemented
from src.data_preprocessing import preprocess_data

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 12
dataset_dir = '../dataset'

# Data augmentation
data_augmentation = tf.keras.Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomRotation(0.3),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomContrast(0.2),
])

# Build model
def build_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze the base_model. Set the bottom layers to be un-trainable
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocessing.Rescaling(1./127.5, offset=-1)(x)  # Rescale input values to [-1, 1]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# Compile model
def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
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

# Callbacks
callbacks = [
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
]

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.2f}")
print(f"Test precision: {test_precision:.2f}")
print(f"Test recall: {test_recall:.2f}")


# Display predictions for the test dataset
display_predictions(test_dataset, model)
# After model evaluation
# Assuming binary classification and test_dataset is properly batched
# Generate predictions
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
y_pred = np.round(np.concatenate([model.predict(images) for images, _ in test_dataset])).astype(int)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Pneumonia'], normalize=True)

# Calculate and print F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")

plt.show()