seed = 123
import numpy as np

np.random.seed(seed)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from dataset.data_loader import get_data
from tensorflow.keras.layers import Dropout
from util.augmentors import mixup, cutmix, cutout, single_cutout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


def setup_model():
    # set seeds to see actual improvements
    tf.random.set_seed(seed)
    tf.random.uniform([1], seed=seed)  # doesn't work?
    labels = ['NORMAL', 'PNEUMONIA']

    # Define parameters for our network
    batch_size = 64
    epochs = 20
    img_height = 150
    img_width = 150
    img_dims = (img_height, img_width)
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.2,
                                  patience=5, min_lr=0.00001, cooldown=3, verbose=1)

    augmentation = 'cutmix'

    alpha = 1
    num_holes = 5

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_data(img_dims, labels)
    total_train = len(train_x)
    total_val = len(val_x)

    if augmentation == 'mixup':
        train_x, train_y = mixup(train_x, train_y, alpha, show_sample=False)
    if augmentation == 'cutmix':
        train_x, train_y = cutmix(train_x, train_y, alpha, show_sample=False)
    if augmentation == 'cutout':
        train_x, train_y = cutout(train_x, train_y, n_holes=num_holes, show_sample=False)

    gen = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        # preprocessing_function=single_cutout
    )

    gen.fit(train_x, seed=seed)

    train_gen = gen.flow(train_x, train_y, batch_size, seed=seed)
    val_gen = gen.flow(val_x, val_y, batch_size, seed=seed)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(img_height, img_width, 1)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.summary()
    path = 'saved_models/{}_weights'.format(augmentation)

    # model.load_weights(path) # uncomment to load weights from last checkpoint
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath=path, save_weights_only=True, monitor='val_accuracy', mode='max',
                                 save_best_only=True, verbose=1)

    history = model.fit(
        train_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=total_val // batch_size,
        shuffle=False, callbacks=[reduce_lr, checkpoint]

    )
    model.load_weights(path)  # load weights for best epoch

    test_loss, test_score = model.evaluate(test_x, test_y, batch_size=batch_size)
    print("Loss on test set: ", test_loss)
    print("Accuracy on test set: ", test_score)

    plot_model(history)


def plot_model(results):
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    loss = results.history['loss']
    val_loss = results.history['val_loss']

    epochs_range = range(len(acc))

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


if __name__ == '__main__':
    setup_model()
