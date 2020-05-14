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
test_dir = os.path.join('res', 'test')
validation_dir = os.path.join('res', 'validation')
train_covid_dir = os.path.join(train_dir, 'NORMAL')
train_non_covid_dir = os.path.join(train_dir, 'PNEUMONIA')
validation_covid_dir = os.path.join(validation_dir, 'NORMAL')
validation_non_covid_dir = os.path.join(validation_dir, 'PNEUMONIA')


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
    batch_size = 8
    epochs = 8
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # Do some rescaling
    # TODO: Will this work with a combination of grayscale/RGB/RGBA??????????
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary',
                                                               color_mode='grayscale')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary',
                                                                  color_mode='grayscale')
    
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=test_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary',
                                                                  color_mode='grayscale')

    sample_training_images, _ = next(train_data_gen)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
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

    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )

    predictions = model.evaluate(test_data_gen, steps=25)
    print(predictions)

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
    for img, ax in zip(images_arr, axes):
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


def count_image_dims(images):
    min_width, min_height = images[0].size
    max_width = min_width
    max_height = min_height

    for img in images:
        width, height = img.size
        if width > max_width:
            max_width = width

        if width < min_width:
            min_width = width

        if height > max_height:
            max_width = height

        if height < min_height:
            min_height = height

    print('Max width: ', max_width)
    print('Min width: ', min_width)
    print('Max height: ', max_height)
    print('Min height: ', min_height)


if __name__ == '__main__':
    actual_stuff()
