import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


dataset_dir = '../dataset'

def load_dataset(dataset_path, img_size=(224, 224)):
    X = []
    y = []

    categories = ['NORMAL', 'PNEUMONIA']

    for category in categories:
        path = os.path.join(dataset_path, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image /= 255.0

            X.append(image[0])
            y.append(class_num)

    return np.array(X), np.array(y)

def flatten_images(images):
    return images.reshape(images.shape[0], -1)


if __name__ == '__main__':


    # Assuming you have a function to load your dataset
    X, y = load_dataset(os.path.join(dataset_dir, 'train'))  # Load training data
    X = flatten_images(X)  # Flatten images

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)