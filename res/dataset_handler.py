import os
import shutil
import pandas as pd
import re


def extract_covid_chestxray_dataset():
    # Extract only covid cases from the chestxray-dataset, and produce a corresponding metadata file
    working_dir = 'covid-chestxray-dataset'
    # Where we will store our extracted images
    covid_path = os.path.join(working_dir, 'covid_cases')
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

    metadata_csv = pd.read_csv(os.path.join(working_dir, 'metadata.csv'),
                               usecols=['patientid', 'finding', 'date', 'filename', 'view', 'folder'])
    metadata_csv.insert(0, 'New_ID', range(0, len(metadata_csv)))  # Adds unique row identifier

    # Extract only Covid cases, as well as counting skipped and covid cases
    covid_cases, skipped = 0, 0
    res = []
    for (i, row) in metadata_csv.iterrows():
        if re.match('images', row['folder']) and re.match('COVID-19', row['finding']):
            # Copy the Covid images to the directory storing all the covid cases
            orig_path = os.path.join(working_dir, 'images', row["filename"])

            # If the script is run multiple times, this will counteract multiple copies of the same image
            if not os.path.exists(os.path.join(covid_path, row["filename"])):
                shutil.copy2(orig_path, covid_path)

            # Keep track of which images are saved
            covid_cases += 1
            res.append(row['New_ID'])
        else:
            skipped += 1

    print('Covid cases: ', covid_cases)
    print('Skipped cases: ', skipped)

    # Save an updated version of the metadata corresponding to the Covid cases
    metadata_csv.loc[metadata_csv['New_ID'].isin(res)].to_csv(os.path.join(working_dir, 'metadata_covid_cases.csv'))


if __name__ == '__main__':
    extract_covid_chestxray_dataset()

