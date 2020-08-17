import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.data_generators import Submission_dataset
from lib.utils import *

#######################
# Training parameters #
#######################

# Parse script aguments
arg_parser = argparse.ArgumentParser(description="Runs the testing of the deep learning model")
arg_parser.add_argument("systole_model", help="Path to the trained model for systole", type=str)
arg_parser.add_argument("diastole_model", help="Path to the trained model for diastole", type=str)
arg_parser.add_argument("--view", help="Type of view to test the model", type=str, choices=["SAX", "2CH", "4CH"], default="SAX")
arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, 
    default="../preproc1_150x150_bySlices_dataset_allViews/")
arg_parser.add_argument("--pdf_mode", help="The way to compute the PDF", type=str, choices=["cdf", "step"], default="cdf")
arg_parser.add_argument("--systole_mae", help="Mean Average Error for systole in validation", type=float, default=15)
arg_parser.add_argument("--diastole_mae", help="Mean Average Error for diastole in validation", type=float, default=20)
arg_parser.add_argument("--out_path", help="File path to store the submission CSV", type=str, default="submissions/results.csv")
arg_parser.add_argument("--sax_systole_model", help="Path to the SAX model for systole to use for the missing cases", type=str, 
    default="models/checkpoints/preproc1_150x150_bySlices_dataset_full_Systole_WideResNet50_0_Adam-0.001_MSE_DA3_best")
arg_parser.add_argument("--sax_diastole_model", help="Path to the SAX model for diastole to use for the missing cases", type=str,
    default="models/checkpoints/preproc1_150x150_bySlices_dataset_full_Diastole_WideResNet50_0_Adam-0.001_MSE_DA3_best")
arg_parser.add_argument("--sax_systole_mae", help="MAE for systole in validation, for the auxiliar SAX model", type=float, default=17.27)
arg_parser.add_argument("--sax_diastole_mae", help="MAE for diastole in validation, for the auxiliar SAX model", type=float, default=22.55)
args = arg_parser.parse_args()

systole_path = args.systole_model
diastole_path = args.diastole_model
data_path = args.data_path
view = args.view
mode = args.pdf_mode
systole_mae = args.systole_mae
diastole_mae = args.diastole_mae
num_workers = args.workers
selected_gpu = args.gpu
pin_memory = args.pin_mem
out_path = args.out_path

# Check computing device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    # Select a GPU
    device = torch.device(f"cuda:{selected_gpu}")
    print(f"Going to test with the GPU in the slot {selected_gpu} -> device model: {torch.cuda.get_device_name(selected_gpu)}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"Cuda is not available, using {device} instead")

##################
# Data generator #
##################

# Load dataset info
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
# Create test datagen
test_dataset = Submission_dataset(test_df, view=view)
test_datagen = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)

##########################
# Load pretrained models #
##########################

model_systole = torch.load(systole_path)
model_diastole = torch.load(diastole_path)

#################
# Testing phase #
#################

# Move the models to the computing devices
model_systole = model_systole.to(device)
model_diastole = model_diastole.to(device)

print("\n###############\n"\
      +f"# TEST PHASE #\n"\
      + "###############\n")

# Get the CDF for all the cases with the selected view avalilable
df, no_data_cases = submission_regresor(test_datagen, 
        model_systole, 
        model_diastole, 
        device, 
        pin_memory, 
        mode=mode, 
        mae=[systole_mae, diastole_mae])

# Check for missing cases to complete with the SAX based regresion
if len(no_data_cases) > 0:
    # Free memory for the new SAX models
    del model_systole
    del model_diastole
    # Load the SAX models
    model_systole = torch.load(args.sax_systole_model)
    model_diastole = torch.load(args.sax_diastole_model)
    # Move models to computing device
    model_systole = model_systole.to(device)
    model_diastole = model_diastole.to(device)
    systole_mae = args.sax_systole_mae
    diastole_mae = args.sax_diastole_mae
    # Create the dataset for SAX view
    sax_test_dataset = Submission_dataset(test_df, view="SAX")
    sax_test_datagen = DataLoader(sax_test_dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
    # Fill the DataFrame with the SAX models
    df = complete_missing_with_SAX(df,
            no_data_cases,
            sax_test_datagen,
            model_systole,
            model_diastole,
            device,
            pin_memory,
            mode=mode,
            mae=[systole_mae, diastole_mae])

# Save the DataFrame to a CSV file
df.to_csv(out_path, index=False)
