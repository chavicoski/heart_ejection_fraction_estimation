import sys
import os
import re
import pandas as pd
from pydicom import dcmread
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob

# Root data path
dataset_path = "../cardiac_dataset"
# Partitons paths
train_path = os.path.join(dataset_path, "train")
dev_path = os.path.join(dataset_path, "validate")
test_path = os.path.join(dataset_path, "test")
# Partitons data paths
train_data_path = os.path.join(train_path, "train")
dev_data_path = os.path.join(dev_path, "validate")
test_data_path = os.path.join(test_path, "test")
# Partitons CSV paths
train_csv_path = os.path.join(train_path, "train.csv")
dev_csv_path = os.path.join(dev_path, "validate.csv")
test_csv_path = os.path.join(test_path, "solution.csv")
# Load CSV's
train_df = pd.read_csv(train_csv_path)
dev_df = pd.read_csv(dev_csv_path)
test_df = pd.read_csv(test_csv_path)

#################################################
# GO THROUGH ALL THE DATASET TO DO THE ANALYSIS #
#################################################
'''
Info:
- Each patient has N sax folders where each folder should have 30 dicom images of the 30 slices in Short Axis Stack (SAX).

- For the cases with more than 30 dicom slices, there will be a multiple of 30 slices wich means that there are more than one sax adquisition. 
  This cases will be the "special cases" in the code and they need special treatment for loading them.

- The folders with less than 30 slices will be ignored. This cases will be the "ignored_cases" in the code
'''
special_cases = []
ignored_cases = []
adquisitons_count = {"train": 0, "dev": 0, "test": 0}
data_splits = [("train", train_data_path), ("dev", dev_data_path), ("test", test_data_path)]

for split_name, data_path in data_splits:
    for case in tqdm(os.listdir(data_path)):
        for sax_dir in glob.glob(os.path.join(data_path, case, "study/sax_*")):  # Go through each sax folder in the case
            dicom_files = glob.glob(os.path.join(sax_dir, "*.dcm"))  # Get the list of dicom files in the sax folder
            n_dicom_files = len(dicom_files) 

            if n_dicom_files == 30:
                for dicom_f in sorted(dicom_files):
                    dicom_data = dcmread(dicom_f) 

            elif n_dicom_files > 30:
                special_cases.append((sax_dir, n_dicom_files))
                n_adquisitions = int(n_dicom_files / 30)

                # Get the set of dicoms for each adquisition 
                for i in range(1, n_adquisitions + 1):
                    # Get the dicoms of the current sax adquisition
                    adquisition_files = sorted(list(filter(lambda x: x.endswith(f"{i}.dcm"), dicom_files)))

            else:
                ignored_cases.append((sax_dir, n_dicom_files))

             
print("\nIngnored cases:")
for sax_dir, n_dicom_files in ignored_cases:
    print(f"{sax_dir} -> {n_dicom_files} dicom files")

print("\nSpecial cases:")
for sax_dir, n_dicom_files in special_cases:
    print(f"{sax_dir} -> {n_dicom_files} dicom files")
