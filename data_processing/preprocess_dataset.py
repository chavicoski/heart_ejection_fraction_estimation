import sys
sys.path.insert(1, '.')  # To access the libraries
import os
import pandas as pd
import torch
from lib.image_processing import get_patient_slices, preprocess_pipeline1

if sys.argv != 3:
    print("Invalid number of arguments!")
    print("Usage: {sys.argv[0]} <IN_DATA_PATH> <OUT_DATA_PATH>")

out_path = sys.argv[2]
os.makedirs(out_path, exist_ok=True)  # Create output directory

# Root data path
dataset_path = sys.argv[1]
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

#############################
# BUILD LABELS DICTIONARIES #
#############################
train_labels, dev_labels, test_labels = dict(), dict(), dict()
splits_labels = [
        ("train", train_df, train_labels), 
        ("dev", dev_df, dev_labels), 
        ("test", test_df, test_labels)
        ]
for split_name, df, labels_dict in splits_labels:
    if split_name != "test":
        for idx, row in df.iterrows():
            labels_dict[row["Id"]] = [row["Systole"], row["Diastole"]]
    else:  # Test dataframe has diferent format
        for idx, row in df.iterrows():
            case_id = int(row["Id"].split("_")[0])
            if case_id not in labels_dict:
                labels_dict[case_id] = [None, None]
            if row["Id"].endswith("Systole"):
                labels_dict[case_id][0] = row["Volume"]
            elif row["Id"].endswith("Diastole"):
                labels_dict[case_id][1] = row["Volume"]

################################################
# PREPROCESS EACH SAMPLE AND CREATE A CSV FILE #
################################################
splits_data = [
        ("train", train_data_path, train_labels),
        ("dev", dev_data_path, dev_labels),
        ("test", test_data_path, test_labels),
        ]
for split_name, data_path, labels_dict in splits_data:
    os.makedirs(os.path.join(out_path, split_name), exist_ok=True) 
    split_df = pd.DataFrame(columns=["Id", "X_path", "Systole", "Diastole"]) 
