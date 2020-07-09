import sys
sys.path.insert(1, '.')  # To access the libraries
import os
import pandas as pd
import torch
from lib.image_processing import get_patient_slices, preprocess_pipeline0, preprocess_pipeline1
from tqdm import tqdm
import argparse

arg_parser = argparse.ArgumentParser(description="Takes the origin raw data and makes a first preprocessing and stores the images in pytorch tensors (.pt files)")


arg_parser.add_argument("in_data", help="Path to the folder with the data to process", type=str)
arg_parser.add_argument("out_data", help="Path to the new folder to store the processed pytorch tensors", type=str)
arg_parser.add_argument("-p", "--preprocess", help="Preprocess pipeline to apply", choices=[0, 1], type=int, default=1)
arg_parser.add_argument("-f", "--format", help="How to store the samples: bySlices: (timesteps, H, W) or byPatients: (slices, timesteps, H, W)", choices=["bySlices", "byPatients"], type=str, default="bySlices")
arg_parser.add_argument("-th", "--target_height", help="Height of the processed images", type=int, default=150)
arg_parser.add_argument("-tw", "--target_width", help="Width of the processed images", type=int, default=150)
args = arg_parser.parse_args()

pipeline_id = args.preprocess # Get preprocess id
samples_format = args.format  # Get shape format of the samples
target_size = (args.target_height, args.target_width)
# Prepare output folder
out_path = args.out_data
os.makedirs(out_path, exist_ok=True)  # Create output directory
# Root data path
dataset_path = args.in_data
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
        ("validate", dev_df, dev_labels), 
        ("test", test_df, test_labels)
        ]
for split_name, df, labels_dict in splits_labels:
    if split_name != "test":
        for idx, row in df.iterrows():
            labels_dict[int(row["Id"])] = [row["Systole"], row["Diastole"]]
    else:  # Test dataframe has diferent format
        for idx, row in df.iterrows():
            case_id = int(row["Id"].split("_")[0])
            if case_id not in labels_dict:
                labels_dict[case_id] = [None, None]
            if row["Id"].endswith("Systole"):
                labels_dict[case_id][0] = row["Volume"]
            elif row["Id"].endswith("Diastole"):
                labels_dict[case_id][1] = row["Volume"]

print("Partitions size:")
print(f"\t-training: {len(train_labels)} cases")
print(f"\t-validation: {len(dev_labels)} cases")
print(f"\t-test: {len(test_labels)} cases")

################################################
# PREPROCESS EACH SAMPLE AND CREATE A CSV FILE #
################################################
splits_data = [
        ("train", train_data_path, train_labels),
        ("validate", dev_data_path, dev_labels),
        ("test", test_data_path, test_labels),
        ]
for split_name, data_path, labels_dict in splits_data:
    split_path = os.path.join(out_path, split_name)
    os.makedirs(split_path, exist_ok=True) 
    split_df = pd.DataFrame(columns=["Id", "SliceId", "X_path", "Systole", "Diastole"]) 
    df_idx = 0  # Index to insert in the DataFrame
    for case_id, (systole, diastole) in tqdm(labels_dict.items()):
        patient_slices, pix_spacings = get_patient_slices(case_id, split_name)  # Get case images
        if patient_slices is None: continue

        # Apply preprocessing pipeline
        if pipeline_id == 0:
            preproc_patient = preprocess_pipeline0(patient_slices, target_size=target_size)  
        elif pipeline_id == 1:
            preproc_patient = preprocess_pipeline1(patient_slices, pix_spacings, target_size=target_size)  
        
        if samples_format == "bySlices":
            # Save each slice image (timesteps, H, W) in a different tensor
            for slice_idx in range(preproc_patient.shape[0]):
                slice_tensor = torch.from_numpy(preproc_patient[slice_idx]).float()  # Create pytorch tensor
                save_path = os.path.join(split_path, f"{case_id}_{slice_idx}.pt")
                torch.save(slice_tensor, save_path)  # Store the pytorch tensor
                split_df.loc[df_idx] = [case_id, slice_idx, save_path, systole, diastole]  # Store the case info in the dataframe
                df_idx += 1
        else:  # samples_format = "byPatients"
            # Save all the slices of a patient (slices, timesteps, H, W) in the same tensor
            patient_tensor = torch.from_numpy(preproc_patient).float()  # Create pytorch tensor
            save_path = os.path.join(split_path, f"{case_id}.pt")
            torch.save(patient_tensor, save_path)  # Store the pytorch tensor
            split_df.loc[df_idx] = [case_id, save_path, systole, diastole]  # Store the case info in the dataframe
            df_idx += 1
    
    split_df.to_csv(os.path.join(out_path, f"{split_name}.csv"), index=False)  # Save the csv of the split
