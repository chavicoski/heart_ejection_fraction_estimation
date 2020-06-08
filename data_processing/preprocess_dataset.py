import sys
sys.path.insert(1, '.')  # To access the libraries
import os
import pandas as pd
import torch
from lib.image_processing import get_patient_slices, preprocess_pipeline1
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Invalid number of arguments!")
    print(f"Usage: {sys.argv[0]} <IN_DATA_PATH> <OUT_DATA_PATH>")
    sys.exit()

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
        ("dev", dev_data_path, dev_labels),
        ("test", test_data_path, test_labels),
        ]
for split_name, data_path, labels_dict in splits_data:
    split_path = os.path.join(out_path, split_name)
    os.makedirs(split_path, exist_ok=True) 
    split_df = pd.DataFrame(columns=["Id", "X_path", "Systole", "Diastole"]) 
    df_idx = 0  # Index to insert in the DataFrame
    for case_id, (systole, diastole) in tqdm(labels_dict.items()):
        patient_slices, pix_spacings = get_patient_slices(case_id, split_name)  # Get case images
        if patient_slices is None: continue
        preproc_patient = preprocess_pipeline1(patient_slices, pix_spacings, target_size=(150, 150))  # Do preprocessing
        patient_tensor = torch.from_numpy(preproc_patient)  # Create pytorch tensor
        save_path = os.path.join(split_path, f"{case_id}.pt")
        torch.save(patient_tensor, save_path)  # Store the pytorch tensor
        split_df.loc[df_idx] = [case_id, save_path, systole, diastole]  # Store the case info in the dataframe
        df_idx += 1
    
    split_df.to_csv(os.path.join(out_path, f"{split_name}.csv"), index=False)  # Save the csv of the split
