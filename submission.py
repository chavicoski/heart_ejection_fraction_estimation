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
arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, default="../preproc1_150x150_bySlices_dataset_full/")
args = arg_parser.parse_args()

systole_path = args.systole_model
diastole_path = args.diastole_model
data_path = args.data_path
num_workers = args.workers
selected_gpu = args.gpu
pin_memory = args.pin_mem

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
test_dataset = Submission_dataset(test_df)
test_datagen = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)

##########################
# Load pretrained models #
##########################

model_systole = torch.load(systole_path)
model_diastole = torch.load(diastole_path)

#################
# Testing phase #
#################

# Get loss function
criterion = nn.MSELoss()

# Move the models to the computing devices
model_systole = model_systole.to(device)
model_diastole = model_diastole.to(device)

print("\n###############\n"\
      +f"# TEST PHASE #\n"\
      + "###############\n")

submission_regresor(test_datagen, model_systole, model_diastole, criterion, device, pin_memory)
