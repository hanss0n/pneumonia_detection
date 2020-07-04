import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.layers import Dropout

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For ease of use
# TODO: should probably be moved to get_data function
path = 'dataset/kaggle/chest_xray/'
normal = 'NORMAL'
pneumonia = 'PNEUMONIA'
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')
print(test_dir)
validation_dir = os.path.join(path, 'val')
train_normal_dir = os.path.join(train_dir, normal)
train_pneumonia_dir = os.path.join(train_dir, pneumonia)
validation_normal_dir = os.path.join(validation_dir, normal)
validation_pneumonia_dir = os.path.join(validation_dir, pneumonia)
test_normal_dir = os.path.join(test_dir, normal)
test_pneumonia_dir = os.path.join(test_dir, pneumonia)


def summarize_dataset(verbose=True):
    # Give an overview of what data we have
    num_class1_tr = len(os.listdir(train_normal_dir))
    num_class2_tr = len(os.listdir(train_pneumonia_dir))

    num_class1_val = len(os.listdir(validation_normal_dir))
    num_class2_val = len(os.listdir(validation_pneumonia_dir))

    num_class1_test = len(os.listdir(test_normal_dir))
    num_class2_test = len(os.listdir(test_pneumonia_dir))

    total_train = num_class1_tr + num_class2_tr
    total_val = num_class1_val + num_class2_val
    total_test = num_class1_test + num_class2_test
    if verbose:
        print('total training ', os.path.basename(os.path.normpath(train_normal_dir)), ' images: ', num_class1_tr)
        print('total training ', os.path.basename(os.path.normpath(train_pneumonia_dir)), ' images: ', num_class2_tr)
        print('total validation ', os.path.basename(os.path.normpath(validation_normal_dir)), ' images: ',
              num_class1_val)
        print('total validation ', os.path.basename(os.path.normpath(validation_pneumonia_dir)), ' images: ',
              num_class2_val)
        print('total testing ', os.path.basename(os.path.normpath(test_normal_dir)), ' images: ', num_class1_test)
        print('total testing ', os.path.basename(os.path.normpath(test_pneumonia_dir)), ' images: ', num_class2_test)
        print('-----------------------------------------------')
        print("Total training images: ", total_train)
        print("Total validation images: ", total_val)
        print("Total test images: ", total_test)
    return total_train, total_val, total_test


labels = ['PNEUMONIA', 'NORMAL']
img_size = 150


def load_data(data_dir, img_dims):
    img_height, img_width = img_dims
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    data = np.array(data)
    train_x = []
    train_y = []
    for img, label in data:
        train_x.append(img)
        train_y.append(label)

    train_x = np.array(train_x) / 255
    train_x = train_x.reshape(-1, img_height, img_width, 1)
    train_y = np.array(train_y)
    return train_x, train_y


def setup_model():
    # set seeds to see actual improvements
    tf.random.set_seed(5)
    tf.random.uniform([1], seed=5)  # doesn't work?

    # Extract the metadata of our dataset
    # TODO: redundant method
    total_train, total_val, total_test = summarize_dataset(verbose=True)

    # Define parameters for our network
    batch_size = 16
    epochs = 12
    img_height = 150
    img_width = 150
    img_dims = (img_height, img_width)

    # 0.7628205418586731
    train_x, train_y = load_data(train_dir, img_dims)
    val_x, val_y = load_data(validation_dir, img_dims)
    test_x, test_y = load_data(test_dir, img_dims)

    gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    gen.fit(train_x)

    train_gen = gen.flow(train_x, train_y, batch_size)
    val_gen = gen.flow(val_x, val_y, batch_size)

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

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=total_val // batch_size
    )

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
