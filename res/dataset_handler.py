import os
import shutil
import pandas as pd
import re

train = 'train'
validation = 'validation'


def setup_directory_structure():
    # Setup the structure below:
    #   train -
    #           covid
    #           non_covid
    #   validation -
    #           covid
    #           non_covid
    create_directory('train')
    create_directory(os.path.join('train', 'covid'))
    create_directory(os.path.join('train', 'non_covid'))
    create_directory('validation')
    create_directory(os.path.join('validation', 'covid'))
    create_directory(os.path.join('validation', 'non_covid'))


def summarize_final_datasets():
    # Give an overview of the contents of the top level datasets
    train_covid = len(os.listdir(os.path.join(train, 'covid')))
    train_non_covid = len(os.listdir(os.path.join(train, 'non_covid')))
    train_tot = train_covid + train_non_covid

    validation_covid = len(os.listdir(os.path.join(validation, 'covid')))
    validation_non_covid = len(os.listdir(os.path.join(validation, 'non_covid')))
    validation_tot = validation_covid + validation_non_covid

    print('Total number of images in training set: ', train_tot)
    print('Covid cases: ', train_covid)
    print('Non covid cases: ', train_non_covid)
    print('-----------------------------------------------')
    print('Total number of images in validation set: ', validation_tot)
    print('Covid cases: ', validation_covid)
    print('Non covid cases: ', validation_non_covid)


def extract_covid_chestxray_dataset():
    # Extract images from the covid-chestxray-dataset and place them in folders /covid and /non-covid
    # Create corresponding metadata files as well
    dataset_dir = 'covid-chestxray-dataset'
    # Where we will store our extracted images
    covid_path = os.path.join(dataset_dir, 'covid')
    non_covid_path = os.path.join(dataset_dir, 'non_covid')

    # Create the directories where the images will be stored
    create_directory(covid_path)
    create_directory(non_covid_path)

    # Read metadata or the /images folder
    metadata_csv = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'),
                               usecols=['patientid', 'finding', 'date', 'filename', 'view', 'folder'])

    # Add unique row identifier
    metadata_csv.insert(0, 'New_ID', range(0, len(metadata_csv)))

    # Extract only valid images, as well as counting non-covid and covid cases
    covid_cases, non_covid_cases, skipped = 0, 0, 0
    covid = []
    non_covid = []
    for (i, row) in metadata_csv.iterrows():
        if re.match('images', row['folder']) and re.match('COVID-19', row['finding']):
            # Copy the Covid images to the directory storing all the covid cases
            orig_path = os.path.join(dataset_dir, 'images', row["filename"])

            # If the script is run multiple times, this will counteract multiple copies of the same image
            if not os.path.exists(os.path.join(covid_path, row["filename"])):
                shutil.copy2(orig_path, covid_path)

            # Keep track of which images are saved
            covid_cases += 1
            covid.append(row['New_ID'])
        else:
            if re.match('volumes', row['folder']):
                skipped += 1
            else:
                orig_path = os.path.join(dataset_dir, 'images', row["filename"])
                if not os.path.exists(os.path.join(non_covid_path, row["filename"])):
                    shutil.copy2(orig_path, non_covid_path)
                non_covid_cases += 1
                non_covid.append(row['New_ID'])

    print('Covid cases: ', covid_cases)
    print('Non Covid cases: ', non_covid_cases)
    print('Skipped: ', skipped)
    print('-----------------------------------------------')

    # Save an updated version of the metadata corresponding to the Covid cases
    metadata_csv.loc[metadata_csv['New_ID'].isin(covid)].to_csv(os.path.join(dataset_dir, 'metadata_covid_cases.csv'))
    metadata_csv.loc[metadata_csv['New_ID'].isin(non_covid)].to_csv(os.path.join(dataset_dir, 'metadata_non_covid.csv'))


def partition_covid_chestxray():
    # Partition the chestxray dataset into training and validation sets stored at top level res/train and res/validation
    dataset_dir = 'covid-chestxray-dataset'

    # TODO: For now we will use a fifty fifty split
    partition_images(0.5, dataset_dir, 'covid')
    partition_images(0.5, dataset_dir, 'non_covid')


def partition_images(train_size, src, classification):
    # Copies a portion of the images in src to res/train/classification and res/validation/classification
    # TODO: Randomize which images --> train?
    data_path = os.path.join(src, classification)
    paths = os.listdir(data_path)
    size = int(train_size * len(paths))
    train_paths = paths[:size]
    validation_paths = paths[size:]
    copy_images2(train_paths, data_path, os.path.join(train, classification))
    copy_images2(validation_paths, data_path, os.path.join(validation, classification))


def copy_images2(paths, src, dest):
    for filename in paths:
        copy_img2(filename, src, dest)


def copy_img2(filename, src, dest):
    if not os.path.exists(os.path.join(dest, filename)):
            shutil.copy2(os.path.join(src, filename), dest)


def create_directory(path):
    # Create a directory at path, given that it doesn't already exist
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
    else:
        print("Directory %s already exists" % path)


if __name__ == '__main__':
    setup_directory_structure()
    extract_covid_chestxray_dataset()
    partition_covid_chestxray()
    summarize_final_datasets()
