import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from PIL import Image
import re


def load_tutorial_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # REPLACE WITH COVID SHIT
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', # Replace with covid, notcovid
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names


def preprocess_data(train_images, train_labels, test_images, test_labels, class_names):
    # scale values to 0 and 1 before feeding to neural network
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=0) # TODO: Fix before commit

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    probability_model = keras.Sequential([model,
                                          keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)


def load_data():
    data = pd.read_csv(os.path.join("res", 'metadata.csv'))
    images = load_images(data.filename)
    labels = load_labels(data.finding)

    # We also have to transform our images to numpy arrays
    images_as_arrays = [np.array(img) for img in images]

    # TODO: Split these two into training + testing sets. For now it's a 50/50 split
    half = int(len(data)/2)
    train_images = images_as_arrays[:half]
    train_labels = labels[:half]
    test_images = images_as_arrays[half:]
    test_labels = labels[half:]
    class_names = ["COVID-19", "NOT COVID-19"] # TODO: Improve naming

    return train_images, train_labels, test_images, test_labels, class_names


def load_images(paths):
    images = [Image.open(os.path.join("res", "images", path)) if not re.match(".*\.gz", path) else None for path in paths]
    # TODO:
    # Currently isn't looking at all 360 images since they are not stored in res/images and needs to be downloaded
    # These paths are stored as .gz files, and the above will filter them out, replacing them with None for now

    return images


def load_labels(labels):
    # As we are only interested in Covid-19 right now, we will compress every non-finding of Covid-19 to the same label
    # Covid = 0, Not Covid = 1
    trimmed_labels = [0 if re.match("COVID-19", label) else 1 for label in labels]
    return trimmed_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, class_names = load_tutorial_data()
    preprocess_data(train_images, train_labels, test_images, test_labels, class_names)+

