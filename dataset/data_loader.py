import kaggle
import os
import random
import shutil

def load_kaggle():
    if not os.path.exists(os.path.join('kaggle', 'chest_xray')):
        dataset = 'paultimothymooney/chest-xray-pneumonia'
        target = 'kaggle'
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)
    else:
        print('The Kaggle dataset is already downloaded')

def re_partition_kaggle():
    num_to_move = 200
    random.seed(1337)
    train_dir = os.path.join('kaggle', 'chest_xray', 'train')
    val_dir = os.path.join('kaggle', 'chest_xray', 'val')

    # Move images for both classes
    train_2_val('NORMAL', train_dir, val_dir, num_to_move)
    train_2_val('PNEUMONIA', train_dir, val_dir, num_to_move)


def train_2_val(classification, train, val, num_to_move):
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

if __name__ == '__main__':
    load_kaggle()
    # re_partition_kaggle()
