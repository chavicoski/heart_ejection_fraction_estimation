import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.data_generators import Ensemble_submission_dataset
from lib.utils import *

#######################
# Training parameters #
#######################

# Parse script aguments
arg_parser = argparse.ArgumentParser(description="Runs the testing of the deep learning model")
arg_parser.add_argument("systole_sax_model", help="Path to the SAX trained model for systole", type=str)
arg_parser.add_argument("diastole_sax_model", help="Path to the SAX trained model for diastole", type=str)
arg_parser.add_argument("systole_2ch_model", help="Path to the 2CH trained model for systole", type=str)
arg_parser.add_argument("diastole_2ch_model", help="Path to the 2CH trained model for diastole", type=str)
arg_parser.add_argument("systole_4ch_model", help="Path to the 4CH trained model for systole", type=str)
arg_parser.add_argument("diastole_4ch_model", help="Path to the 4CH trained model for diastole", type=str)

arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, 
    default="../preproc1_150x150_bySlices_dataset_allViews/")
arg_parser.add_argument("--pdf_mode", help="The way to compute the PDF", type=str, choices=["cdf", "step"], default="cdf")
arg_parser.add_argument("--systole_mae", help="Mean Average Error for systole in validation", type=float, default=15)
arg_parser.add_argument("--diastole_mae", help="Mean Average Error for diastole in validation", type=float, default=20)
arg_parser.add_argument("--out_path", help="File path to store the submission CSV", type=str, default="submissions/results.csv")
args = arg_parser.parse_args()

systole_sax_path = args.systole_sax_model
diastole_sax_path = args.diastole_sax_model
systole_2ch_path = args.systole_2ch_model
diastole_2ch_path = args.diastole_2ch_model
systole_4ch_path = args.systole_4ch_model
diastole_4ch_path = args.diastole_4ch_model
data_path = args.data_path
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
test_dataset = Ensemble_submission_dataset(test_df)
test_datagen = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)

#################
# Testing phase #
#################

print("\n###############\n"\
      +f"# TEST PHASE #\n"\
      + "###############\n")

# Get the CDF for all the cases with the selected view avalilable
df = submission_ensemble(systole_sax_path, diastole_sax_path,
                         systole_2ch_path, diastole_2ch_path,
                         systole_4ch_path, diastole_4ch_path,
                         test_datagen, device, pin_memory, 
                         mode=mode,
                         mae=[systole_mae, diastole_mae])

# Save the DataFrame to a CSV file
df.to_csv(out_path, index=False)
