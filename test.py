import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.data_generators import Cardiac_dataset
from lib.utils import *

#######################
# Training parameters #
#######################

# Parse script aguments
arg_parser = argparse.ArgumentParser(description="Runs the testing of the deep learning model")
arg_parser.add_argument("target_label", help="Target label to test", type=str, choices=["Systole", "Diastole"])
arg_parser.add_argument("trained_model", help="Path to the trained model", type=str)
arg_parser.add_argument("--view", help="Type of view to train the model for", type=str, choices=["SAX", "2CH", "4CH"], default="SAX")
arg_parser.add_argument("-bs", "--batch_size", help="Samples per training batch", type=int, default=128)
arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--multi_gpu", help="Use all the available GPU's for training", action="store_true", default=False)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, default="../preproc1_150x150_bySlices_dataset_full/")
arg_parser.add_argument("-loss", "--loss_function", help="Loss function to optimize during training", type=str, choices=["MSE", "MAE"], default="MSE")
args = arg_parser.parse_args()

data_path = args.data_path
dataset_name = get_dataset_name(data_path)
view = args.view
loss_function = args.loss_function
batch_size = args.batch_size
num_workers = args.workers
selected_gpu = args.gpu
multi_gpu = args.multi_gpu
pin_memory = args.pin_mem
target_label = args.target_label
model_path = args.trained_model

# Check computing device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    # Select a GPU
    device = torch.device(f"cuda:{selected_gpu}")
    if n_gpus > 1 and multi_gpu:
        print("Going to use multi GPU training")
        print(f"{n_gpus} GPU's available:")
        for gpu_idx in range(n_gpus):
            print(f"\t-At device cuda:{gpu_idx} -> device model = {torch.cuda.get_device_name(gpu_idx)}")
    else:
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
test_dataset = Cardiac_dataset(test_df, target_label, view=view)
test_datagen = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

#########################
# Load pretrained model #
#########################

model = torch.load(model_path)
# Print model architecture
print(f"Model architecture:\n {model} \n")

#################
# Testing phase #
#################

# Get loss function
if loss_function == "MSE":
    criterion = nn.MSELoss()
elif loss_function == "MAE":
    criterion = nn.L1Loss()
else:
    print(f"Loss function {loss_function} is not valid!")
    sys.exit()

# Prepare multi-gpu training if enabled
if multi_gpu and n_gpus > 1 :
    print("Preparing multi-gpu training...")
    model = nn.DataParallel(model)

# Move the model to the computing devices
model = model.to(device)

# Print training header
print("\n###############\n"\
      +f"# TEST PHASE #\n"\
      + "###############\n")

test_loss, test_diff = test_regresor(test_datagen, model, criterion, device, pin_memory)

print(f"Test results: loss: {test_loss:.5f} - diff: {test_diff:.4f}ml")
