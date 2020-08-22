import kaggle
import os
import random
import shutil
import cv2
import numpy as np
import tqdm
import sys
import image

# loads dataset from kaggle
def load_kaggle():
    if not os.path.exists(os.path.join('kaggle', 'chest_xray')):
        dataset = 'paultimothymooney/chest-xray-pneumonia'
        target = 'kaggle'
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)
    else:
        print('The Kaggle dataset is already downloaded')

def re_partition_kaggle():
    num_to_move = 42
    random.seed(1337)
    test_dir = os.path.join('kaggle', 'chest_xray', 'test')
    val_dir = os.path.join('kaggle', 'chest_xray', 'val')

    # Move images for both classes
    test_2_val('NORMAL', test_dir, val_dir, num_to_move)
    test_2_val('PNEUMONIA', test_dir, val_dir, num_to_move)


def test_2_val(classification, train, val, num_to_move):
    if len(os.listdir(os.path.join(val, classification))) == 8:
        images = os.listdir(os.path.join(train, classification))
        idx = random.sample(range(0, len(images)), num_to_move)
        images_to_move = [images[i] for i in idx]
        move_images(os.path.join(train, classification), os.path.join(val, classification), images_to_move)
    else:
        print('The validation set already has ', len(os.listdir(os.path.join(val, classification))), ' entries')


def move_images(src, dest, images):
    for filename in images:
        shutil.move(os.path.join(src, filename), dest)


# DISCLAIMER: code from: https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
def load_and_preprocess_data(data_dir, img_dims, labels):
    img_height, img_width = img_dims
    data = []

    dir_len = sum([len(files) for r, d, files in os.walk(data_dir)])
    print("Processing {} images in {}.".format(dir_len, data_dir))

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in tqdm.tqdm(os.listdir(path), desc='{}'.format(label), file=sys.stdout):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_height, img_width))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])

            except Exception as e:
                print(e)
    data = np.array(data,dtype=object)
    set_x = []
    set_y = []
    for img, label in data:
        set_x.append(img)
        set_y.append(label)

    set_x = np.array(set_x) / 255
    set_x = set_x.reshape(-1, img_height, img_width, 1)
    set_y = np.array(set_y)
    return set_x, set_y


def get_data(img_dims, labels,process):
    preprocessed_path = 'dataset/preprocessed/'
    if process:
        # if not os.path.isdir(preprocessed_path):
        #     raw_path = 'dataset/kaggle/chest_xray/'
        #     if not os.path.isdir(raw_path):
        #         load_kaggle()

        if  os.path.exists(preprocessed_path + 'train_x.npy'):
            os.remove(preprocessed_path + 'train_x.npy')
            os.remove(preprocessed_path + 'train_y.npy')
            os.remove(preprocessed_path + 'val_x.npy')
            os.remove(preprocessed_path + 'val_y.npy')
            os.remove(preprocessed_path + 'test_x.npy')
            os.remove(preprocessed_path + 'test_y.npy')

        raw_path = 'dataset/kaggle/chest_xray/'

        train_dir = os.path.join(raw_path, 'train')
        test_dir = os.path.join(raw_path, 'test')
        validation_dir = os.path.join(raw_path, 'val')

        (train_x, train_y) = load_and_preprocess_data(train_dir, img_dims, labels)
        (val_x, val_y) = load_and_preprocess_data(validation_dir, img_dims, labels)
        (test_x, test_y) = load_and_preprocess_data(test_dir, img_dims, labels)

        np.save(preprocessed_path + 'train_x', train_x)
        np.save(preprocessed_path + 'train_y', train_y)
        np.save(preprocessed_path + 'val_x', val_x)
        np.save(preprocessed_path + 'val_y', val_y)
        np.save(preprocessed_path + 'test_x', test_x)
        np.save(preprocessed_path + 'test_y', test_y)

    train_x = np.load(preprocessed_path + 'train_x.npy')
    train_y = np.load(preprocessed_path + 'train_y.npy')
    val_x = np.load(preprocessed_path + 'val_x.npy')
    val_y = np.load(preprocessed_path + 'val_y.npy')
    test_x = np.load(preprocessed_path + 'test_x.npy')
    test_y = np.load(preprocessed_path + 'test_y.npy')

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def reset_Models():
    folders = ['saved_models/','plots/']
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))




def count_color_modes(images):
    grey_scale, rgb, rgba, i = 0, 0, 0, 0
    for im in images:
        img = Image.open(im)
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
    load_kaggle()
    re_partition_kaggle()
