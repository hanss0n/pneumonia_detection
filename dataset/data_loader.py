import kaggle
import os
import random
import shutil
import re


def load_kaggle(dataset, target, subdir=''):
    """
    Download a dataset from kaggle.com

    :param dataset: The dataset to download
    :param target: Where to download it to
    :param subdir: Default empty. Optional string which specifies the path from target to the actual data.
                   Ex. if there is a subdirectory to the target on kaggle.com, target/subdir/images

    :return the path to the directory containing the images. That is, the following paths exists: path/test,
            path/train, path/val
    """
    data_path = os.path.join(target, subdir)
    if not os.path.exists(data_path):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)
    else:
        print('The Kaggle dataset is already downloaded')
    return data_path


def increase_val_set():
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


def collect_all_filepaths(src, subsets, classifications):
    """
    Collect all filepaths for the images in src/subset/classification, for each of the subsets and classifications

    :param src: The top directory containing the different subsets
    :param subsets: The different subsets, i.e. training, test and validation sets for example
    :param classifications: The different classifications

    :return: All filepaths to the images
    """
    filepaths = []
    for subset in subsets:
        for classification in classifications:
            subdir = os.path.join(src, subset, classification)
            for img_path in os.listdir(subdir):
                filepaths.append(os.path.join(subdir, img_path))
    return filepaths


def partition_dataset(subsets, classifications, train, val, test, images):
    """
    Partition all the images into the sets listed in subsets, with a split
    specified by the 3 arguments train, val and test. The user supplies the
    splits, which must sum to 100.

    :param subsets: The sets to partition into, ex train, val and test sets
    :param classifications: The classifications of the images
    :param train: The split intended for training data
    :param val: The split intended for validation data
    :param test: The split intended for test data
    :param images: A list of images (paths to images) to partition
    """
    random.seed(1337)
    for subset in subsets:
        if not os.path.exists(subset):
            os.mkdir(subset)
        for classification in classifications:
            if not os.path.exists(os.path.join(subset, classification)):
                os.mkdir(os.path.join(subset, classification))

    if train + val + test != 100:
        print('Please supply a valid split (train + val + test = 100) \n'
              'Currently the split is: \n'
              'Train =', train, '\n'
              'Val =', val, '\n'
              'Test =', test)
    else:
        subset_regex = '(' + ('|'.join(subsets)) + ')'
        for img in images:
            partition = random.randrange(100)
            if partition < train:
                new_set = 'train'
            elif train <= partition < train + val:
                new_set = 'val'
            else:
                new_set = 'test'
            new_path = re.sub('.+' + os.path.sep + subset_regex, new_set, img)
            shutil.copy(img, new_path)


if __name__ == '__main__':
    dataset_path = load_kaggle('paultimothymooney/chest-xray-pneumonia', 'kaggle', 'chest_xray')
    subsets = ['test', 'train', 'val']
    classifications = ['NORMAL', 'PNEUMONIA']

    img_paths = collect_all_filepaths(dataset_path, subsets, classifications)
    partition_dataset(subsets, classifications, 70, 15, 15, img_paths)

