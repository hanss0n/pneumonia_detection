import os
import shutil
import pandas as pd
import re


def extract_covid_chestxray_dataset():
    # Extract only covid cases from the chestxray-dataset, and produce a corresponding metadata file
    working_dir = 'covid-chestxray-dataset'
    # Where we will store our extracted images
    covid_path = os.path.join(working_dir, 'covid_cases')
    non_covid_path = os.path.join(working_dir, 'non_covid')
    # If it hasn't already been created, do so
    if not os.path.exists(covid_path):
        try:
            os.mkdir(covid_path)
        except OSError:
            print("Creation of the directory %s failed" % covid_path)
        else:
            print("Successfully created the directory %s " % covid_path)
    else:
        print("Directory %s already exists" % covid_path)

    if not os.path.exists(non_covid_path):
        try:
            os.mkdir(non_covid_path)
        except OSError:
            print("Creation of the directory %s failed" % non_covid_path)
        else:
            print("Successfully created the directory %s " % non_covid_path)
    else:
        print("Directory %s already exists" % non_covid_path)

    metadata_csv = pd.read_csv(os.path.join(working_dir, 'metadata.csv'),
                               usecols=['patientid', 'finding', 'date', 'filename', 'view', 'folder'])
    metadata_csv.insert(0, 'New_ID', range(0, len(metadata_csv)))  # Adds unique row identifier

    # Extract only Covid cases, as well as counting non- and covid cases
    covid_cases, non_covid_cases, skipped = 0, 0, 0
    covid = []
    non_covid = []
    for (i, row) in metadata_csv.iterrows():
        if re.match('images', row['folder']) and re.match('COVID-19', row['finding']):
            # Copy the Covid images to the directory storing all the covid cases
            orig_path = os.path.join(working_dir, 'images', row["filename"])

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
                orig_path = os.path.join(working_dir, 'images', row["filename"])
                if not os.path.exists(os.path.join(non_covid_path, row["filename"])):
                    shutil.copy2(orig_path, non_covid_path)
                non_covid_cases += 1
                non_covid.append(row['New_ID'])

    print('Covid cases: ', covid_cases)
    print('Non Covid cases: ', non_covid_cases)
    print('Skipped: ', skipped)

    # Save an updated version of the metadata corresponding to the Covid cases
    metadata_csv.loc[metadata_csv['New_ID'].isin(covid)].to_csv(os.path.join(working_dir, 'metadata_covid_cases.csv'))
    metadata_csv.loc[metadata_csv['New_ID'].isin(non_covid)].to_csv(os.path.join(working_dir, 'metadata_non_covid.csv'))


if __name__ == '__main__':
    extract_covid_chestxray_dataset()

