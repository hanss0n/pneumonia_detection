import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# For ease of use
path = 'dataset/kaggle/chest_xray/'
class1 = 'NORMAL'
class2 = 'PNEUMONIA'
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')
validation_dir = os.path.join(path, 'val')
train_class1_dir = os.path.join(train_dir, class1)
train_class2_dir = os.path.join(train_dir, class2)
validation_class1_dir = os.path.join(validation_dir, class1)
validation_class2_dir = os.path.join(validation_dir, class2)
test_class1_dir = os.path.join(test_dir, class1)
test_class2_dir = os.path.join(test_dir, class2)


def summarize_dataset(verbose=True):
    # Give an overview of what data we have
    num_class1_tr = len(os.listdir(train_class1_dir))
    num_class2_tr = len(os.listdir(train_class2_dir))

    num_class1_val = len(os.listdir(validation_class1_dir))
    num_class2_val = len(os.listdir(validation_class2_dir))

    num_class1_test = len(os.listdir(test_class1_dir))
    num_class2_test = len(os.listdir(test_class2_dir))

    total_train = num_class1_tr + num_class2_tr
    total_val = num_class1_val + num_class2_val
    total_test = num_class1_test + num_class2_test
    if verbose:
        print('total training ', os.path.basename(os.path.normpath(train_class1_dir)), ' images: ', num_class1_tr)
        print('total training ', os.path.basename(os.path.normpath(train_class2_dir)), ' images: ', num_class2_tr)
        print('total validation ', os.path.basename(os.path.normpath(validation_class1_dir)), ' images: ',
              num_class1_val)
        print('total validation ', os.path.basename(os.path.normpath(validation_class2_dir)), ' images: ',
              num_class2_val)
        print('total testing ', os.path.basename(os.path.normpath(test_class1_dir)), ' images: ', num_class1_test)
        print('total testing ', os.path.basename(os.path.normpath(test_class2_dir)), ' images: ', num_class2_test)
        print('-----------------------------------------------')
        print("Total training images: ", total_train)
        print("Total validation images: ", total_val)
        print("Total test images: ", total_test)
    return total_train, total_val, total_test


def setup_model():
    # Extract the metadata of our dataset
    total_train, total_val, total_test = summarize_dataset(verbose=True)

    # Define parameters for our network
    batch_size = 16
    epochs = 15
    img_height = 150
    img_width = 150

    # Do some rescaling
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode='binary',
                                                               color_mode='grayscale')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(img_height, img_width),
                                                                  class_mode='binary',
                                                                  color_mode='grayscale')

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=test_dir,
                                                             target_size=(img_height, img_width),
                                                             class_mode='binary',
                                                             color_mode='grayscale')

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


if __name__ == '__main__':
    setup_model()
