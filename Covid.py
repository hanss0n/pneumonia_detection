import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from PIL import Image
import re


train_dir = os.path.join('res', 'train')
validation_dir = os.path.join('res', 'validation')
train_covid_dir = os.path.join(train_dir, 'covid')
train_non_covid_dir = os.path.join(train_dir, 'non_covid')
validation_covid_dir = os.path.join(validation_dir, 'covid')
validation_non_covid_dir = os.path.join(validation_dir, 'non_covid')


def load_tutorial_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # REPLACE WITH COVID SHIT
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',  # Replace with covid_cases, notcovid
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
    path_chestxray = os.path.join('res', 'covid-chestxray-dataset')
    metadata_csv = pd.read_csv(os.path.join(path_chestxray, 'metadata.csv'),
                               usecols=['patientid', 'finding', 'date', 'filename', 'view', 'folder'])

    # Add unique row identifier
    metadata_csv.insert(0, 'New_ID', range(0, len(metadata_csv)))

    # Extract only images in the /images folder, as well as updating the metadata accordingly
    covid_cases, skipped = 0, 0
    usable_rows = []
    for (i, row) in metadata_csv.iterrows():
        if re.match('images', row['folder']):
            covid_cases += 1
            usable_rows.append(row['New_ID'])
        else:
            # There are 21 images in 'volume', we skip them for now
            skipped += 1
    # Update metadata
    usable_metadata = metadata_csv.loc[metadata_csv['New_ID'].isin(usable_rows)]

    # Store all the images we want to use
    images = [Image.open(os.path.join(path_chestxray, 'images', path)) for path in usable_metadata.filename]
    # We also have to transform our images to numpy arrays
    # This takes a year...
    images_as_arrays = [np.array(img) for img in images]

    # Extract all the labels for the images we are using
    # Covid = 0, Not Covid = 1
    labels = [0 if re.match('COVID-19', label) else 1 for label in usable_metadata.finding]

    # TODO: Split these two into training + testing sets. For now it's a 50/50 split
    half = int(len(usable_metadata) / 2)
    train_images = images_as_arrays[:half]
    train_labels = labels[:half]
    test_images = images_as_arrays[half:]
    test_labels = labels[half:]
    class_names = ['COVID-19', 'NOT COVID-19']

    return train_images, train_labels, test_images, test_labels, class_names


def actual_stuff():
    # Give an overview of what data we have
    num_covid_tr = len(os.listdir(train_covid_dir))
    num_non_covid_tr = len(os.listdir(train_non_covid_dir))

    num_covid_val = len(os.listdir(validation_covid_dir))
    num_non_covid_val = len(os.listdir(validation_non_covid_dir))

    total_train = num_covid_tr + num_non_covid_tr
    total_val = num_covid_val + num_non_covid_val

    print('total training covid images:', num_covid_tr)
    print('total training non_covid images:', num_non_covid_tr)

    print('total validation covid images:', num_covid_val)
    print('total validation non_covid images:', num_non_covid_val)
    print('-----------------------------------------------')
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    # Define parameters for our network
    batch_size = 10
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # Do some rescaling
    # TODO: Will this work with a combination of grayscale/RGB/RGBA??????????
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    sample_training_images, _ = next(train_data_gen)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def count_color_modes(images):
    grey_scale, rgb, rgba, i = 0, 0, 0, 0
    for img in images:
        if img.mode == 'L':
            grey_scale += 1
        if img.mode == 'RGB':
            rgb += 1
        if img.mode == 'RGBA':
            rgba += 1
        i = i + 1

    print('All: ', i)
    print('Grayscale: ', grey_scale)
    print('RGB: ', rgb)
    print('RGBA: ', rgba)


if __name__ == '__main__':
    # train_images, train_labels, test_images, test_labels, class_names = load_data()
    # print(sys.getsizeof(train_images))
    # preprocess_data(train_images, train_labels, test_images, test_labels, class_names)
    actual_stuff()






