import kaggle
import os
import random
import shutil


def load_kaggle(dataset, target, subdir=''):
    """
    Download a dataset from kaggle.com

    :param dataset: The dataset to download
    :param target: Where to download it to
    :param subdir: Default empty. Optional string which specifies the path from target to the actual data.
                   Ex. if there is a subdirectory to the target on kaggle.com, target/subdir/images

    :return the path to the directory containing the images. That is, the path/test, path/train, path/val
    """
    data_path = os.path.join(target, subdir)
    print(data_path)
    if not os.path.exists(data_path):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)
        return data_path
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
    load_kaggle('paultimothymooney/chest-xray-pneumonia', 'kaggle')
    increase_val_set()
