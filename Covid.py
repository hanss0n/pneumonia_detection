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
import shutil


def load_tutorial_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # REPLACE WITH COVID SHIT
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',  # Replace with covid, notcovid
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

    model.fit(train_images, train_labels, epochs=0)  # TODO: Fix before commit

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
    half = int(len(data) / 2)
    train_images = images_as_arrays[:half]
    train_labels = labels[:half]
    test_images = images_as_arrays[half:]
    test_labels = labels[half:]
    class_names = ["COVID-19", "NOT COVID-19"]  # TODO: Improve naming

    return train_images, train_labels, test_images, test_labels, class_names


def load_images(paths):
    images = [Image.open(os.path.join("res", "images", path)) if not re.match(".*\.gz", path) else None for path in
              paths]
    # TODO:
    # Currently isn't looking at all 360 images since they are not stored in res/images and needs to be downloaded
    # These paths are stored as .gz files, and the above will filter them out, replacing them with None for now

    return images


def load_labels(labels):
    # As we are only interested in Covid-19 right now, we will compress every non-finding of Covid-19 to the same label
    # Covid = 0, Not Covid = 1
    trimmed_labels = [0 if re.match('COVID-19', label) else 1 for label in labels]
    return trimmed_labels


def extract_covid_chestxray_dataset():
    working_dir = os.path.join('res', 'covid-chestxray-dataset')
    # Where we will store our extracted images
    covid_path = os.path.join(working_dir, 'covid_cases')

    metadata_csv = pd.read_csv(os.path.join(working_dir, 'metadata.csv'),
                               usecols=['patientid', 'finding', 'date', 'filename', 'view', 'folder'])
    metadata_csv.insert(0, 'New_ID', range(0, len(metadata_csv)))  # Adds unique row identifier

    # Extract only Covid cases, as well as counting skipped and covid cases
    covid_cases, skipped = 0, 0
    res = []
    for (i, row) in metadata_csv.iterrows():
        if re.match('images', row['folder']) and re.match('COVID-19', row['finding']):
            # Copy the Covid images to the directory storing all the covid cases
            orig_path = os.path.join(working_dir, 'images', row["filename"])
            if not os.path.exists(covid_path):
                shutil.copy2(orig_path, covid_path)

            # Keep track of which images are saved
            covid_cases += 1
            res.append(row['New_ID'])
        else:
            skipped += 1

    print('Covid cases: ', covid_cases)
    print('Skipped cases: ', skipped)

    # Save an updated version of the metadata corresponding to the Covid cases
    metadata_csv.loc[metadata_csv['New_ID'].isin(res)].to_csv(os.path.join(working_dir, 'metadata_covid_cases.csv'))


if __name__ == '__main__':
    # preprocess_data(train_images, train_labels, test_images, test_labels, class_names)
    # train_images, train_labels, test_images, test_labels, class_names = load_data()
    extract_covid_chestxray_dataset()
    metadata_csv = pd.read_csv(os.path.join('res', 'covid-chestxray-dataset', 'metadata_covid_cases.csv'))
    print(metadata_csv)


