import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing import image_dataset_from_directory

# For ease of use
# TODO: should probably be moved to get_data function
path = 'dataset/kaggle/chest_xray/'
normal = 'NORMAL'
pneumonia = 'PNEUMONIA'
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')
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


def get_data(batch_size, img_dims, shuffle=False):
    (img_height, img_width) = img_dims
    train_data = image_dataset_from_directory(train_dir, color_mode='grayscale',
                                              image_size=(img_height, img_width), batch_size=batch_size,
                                              shuffle=shuffle, seed=1337)
    val_data = image_dataset_from_directory(validation_dir, color_mode='grayscale',
                                            image_size=(img_height, img_width), batch_size=batch_size, shuffle=False)
    test_data = image_dataset_from_directory(test_dir, color_mode='grayscale',
                                             image_size=(img_height, img_width),
                                             batch_size=batch_size, shuffle=False)
    return train_data, val_data, test_data


def setup_model():
    # set seeds to see actual improvements
    tf.random.set_seed(5)
    tf.random.uniform([1], seed=5)  # doesn't work?

    # Extract the metadata of our dataset
    # TODO: redundant method
    total_train, total_val, total_test = summarize_dataset(verbose=True)

    # Define parameters for our network
    batch_size = 16
    epochs = 1
    img_height = 150
    img_width = 150

    # 0.7628205418586731
    train_data, val_data, test_data = get_data(batch_size, (img_height, img_width), shuffle=True)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
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
        train_data,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data,
        validation_steps=total_val // batch_size
    )

    predictions = model.evaluate(test_data, steps=len(test_data))
    print(predictions)

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
