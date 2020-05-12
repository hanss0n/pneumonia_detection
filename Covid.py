import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


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




if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, class_names = load_tutorial_data()
    preprocess_data(train_images, train_labels, test_images, test_labels, class_names)+

