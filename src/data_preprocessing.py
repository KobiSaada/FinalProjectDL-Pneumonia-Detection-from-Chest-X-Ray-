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

