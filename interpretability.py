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
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing as mp

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
arg_parser.add_argument("--split", help="Data split to take images", type=str, choices=["train", "validate", "test"], default="validate")
arg_parser.add_argument("-out", "--output_path", help="Folder to save the plots of the attributions", type=str, default="attributions_plots")
arg_parser.add_argument("-gifs", "--make_gifs", help="Enable the gifs creation", choices=[0, 1], type=bool, default=True)
arg_parser.add_argument("--nproc_gifs", help="Number of processes to create the gifs in parallel (0 = all cores)", choices=[i for i in range(mp.cpu_count() + 1)], type=int, default=0)
args = arg_parser.parse_args()

systole_path = args.systole_model
diastole_path = args.diastole_model
data_path = args.data_path
out_path = args.output_path
split = args.split
num_workers = args.workers
selected_gpu = args.gpu
pin_memory = args.pin_mem
make_gifs = args.make_gifs
gifs_cores = args.nproc_gifs

# Check computing device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    # Select a GPU
    device = torch.device(f"cuda:{selected_gpu}")
    print(f"\nGoing use the GPU in the slot {selected_gpu} -> device model: {torch.cuda.get_device_name(selected_gpu)}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"\nCuda is not available, using {device} instead")

##################
# Data generator #
##################

# Load dataset info
df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
# Create test datagen
dataset = Submission_dataset(df)
datagen = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)

##########################
# Load pretrained models #
##########################

model_systole = torch.load(systole_path)
model_diastole = torch.load(diastole_path)

# Move the models to the computing devices
model_systole = model_systole.to(device)
model_diastole = model_diastole.to(device)

# Set the inference mode
model_systole = model_systole.eval()
model_diastole = model_diastole.eval()

#########################
# Intepretability phase #
#########################

# Load the sample to analyze
sample = next(iter(datagen))
id_, data, y_systole, y_diastole = sample["ID"], sample["X"], sample["Y_systole"], sample["Y_diastole"]
data = data.to(device, non_blocking=pin_memory)[0]
n_slices = data.size(0)
n_timesteps = data.size(1)
print(f"\nGoing to analyze case {id_.item()} from {split} split:")

# Get the predicted values and error for the full case classification
pred_systole = model_systole(data).mean()
pred_diastole = model_diastole(data).mean()
err_systole = torch.abs(pred_systole - y_systole).item()
err_diastole = torch.abs(pred_diastole - y_diastole).item()

print(f"\nTrue values:\n\tsystole  = {y_systole.item():>6.2f}\n\tdiastole = {y_diastole.item():>6.2f}")
print(f"\nPredicted values:\n\tsystole  = {pred_systole:>6.2f}\n\tdiastole = {pred_diastole:>6.2f}")
print(f"\nAbsolute error:\n\tsystole  = {err_systole:>5.2f}\n\tdiastole = {err_diastole:>5.2f}")

# Integrated gradients algorithm
ig_systole = IntegratedGradients(model_systole)
ig_diastole = IntegratedGradients(model_diastole)

# Noise tunnel (integrated gradients)
noise_tunnel_ig_sys = NoiseTunnel(ig_systole)
noise_tunnel_ig_dias = NoiseTunnel(ig_diastole)

# For the attributions plots
default_cmap = LinearSegmentedColormap.from_list(
        'custom blue',
        [(0, '#ffffff'),
        (0.25, '#000000'),
        (1, '#000000')], N=256)

# Create output folders for the attributions plots
os.makedirs(out_path, exist_ok=True)
case_out_path = os.path.join(out_path, f"case_{id_.item()}")
os.makedirs(case_out_path, exist_ok=True)

# To store the paths to the plots folders for making later the animations
if make_gifs: gifs_folders = []

# Analyze every slice of the case
for slice_id in range(n_slices):

    # Get slice data with shape (1, timesteps, H, W)
    slice_data = data[slice_id].unsqueeze_(0)

    # Get RGB images from the slice
    slice_images = to_RGB_images(slice_data)

    # Compute prediction and error
    pred_systole = model_systole(slice_data).item()
    pred_diastole = model_diastole(slice_data).item()
    err_systole = pred_systole - y_systole.item()
    err_diastole = pred_diastole - y_diastole.item()
    print(f"\nSlice {slice_id}:\n\tsystole_err  = {err_systole:>6.2f}\n\tdiastole_err = {err_diastole:>6.2f}")

    # Create the slice folder structure
    slice_out_path = os.path.join(case_out_path, f"slice_{slice_id}")
    os.makedirs(slice_out_path, exist_ok=True)
    slice_sys_out_path = os.path.join(slice_out_path, f"systole_{err_systole:.2f}")
    os.makedirs(slice_sys_out_path, exist_ok=True)
    slice_dias_out_path = os.path.join(slice_out_path, f"diastole_{err_diastole:.2f}")
    os.makedirs(slice_dias_out_path, exist_ok=True)

    '''Integrated Gradients'''
    # Create folders
    slice_ig_sys_out_path = os.path.join(slice_sys_out_path, f"integrated_gradients")
    os.makedirs(slice_ig_sys_out_path, exist_ok=True)
    slice_ig_dias_out_path = os.path.join(slice_dias_out_path, f"integrated_gradients")
    os.makedirs(slice_ig_dias_out_path, exist_ok=True)

    # Compute attributions
    attributions_ig_systole = ig_systole.attribute(slice_data, n_steps=400, internal_batch_size=1)
    attributions_ig_diastole = ig_diastole.attribute(slice_data, n_steps=400, internal_batch_size=1)

    # Get RGB images from the attributions
    attr_ig_systole_images = to_RGB_images(attributions_ig_systole)
    attr_ig_diastole_images = to_RGB_images(attributions_ig_diastole)

    '''Noise tunnel for integrated gradients'''
    # Create folders
    slice_noise_ig_sys_out_path = os.path.join(slice_sys_out_path, f"noise_tunnel_ig")
    os.makedirs(slice_noise_ig_sys_out_path, exist_ok=True)
    slice_noise_ig_dias_out_path = os.path.join(slice_dias_out_path, f"noise_tunnel_ig")
    os.makedirs(slice_noise_ig_dias_out_path, exist_ok=True)

    # Compute attributions
    attributions_noise_ig_systole = noise_tunnel_ig_sys.attribute(slice_data, n_samples=10, nt_type='smoothgrad_sq', internal_batch_size=1)
    attributions_noise_ig_diastole = noise_tunnel_ig_dias.attribute(slice_data, n_samples=10, nt_type='smoothgrad_sq', internal_batch_size=1)

    # Get RGB images from the attributions
    attr_noise_ig_systole_images = to_RGB_images(attributions_noise_ig_systole)
    attr_noise_ig_diastole_images = to_RGB_images(attributions_noise_ig_diastole)

    # Make plots of the selected slice and timestep
    for t in range(n_timesteps):
        '''Integrated Gradients'''
        fig_sys, ax_sys = viz.visualize_image_attr_multiple(
                attr_ig_systole_images[t],
                slice_images[t],
                ["original_image", "heat_map"],
                ["all", "positive"],
                titles=["Original", "Integrated Gradients"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False)
        fig_dias, ax_dias = viz.visualize_image_attr_multiple(
                attr_ig_diastole_images[t],
                slice_images[t],
                ["original_image", "heat_map"],
                ["all", "positive"],
                titles=["Original", "Integrated Gradients"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False)

        # Save plots
        fig_sys.savefig(os.path.join(slice_ig_sys_out_path, f"ig_systole_step{t:02}_{err_systole:.2f}err.png"))
        fig_dias.savefig(os.path.join(slice_ig_dias_out_path, f"ig_diastole_step{t:02}_{err_diastole:.2f}err.png"))

        '''Noise tunnel for integrated gradients'''
        fig_sys, ax_sys = viz.visualize_image_attr_multiple(
                attr_noise_ig_systole_images[t],
                slice_images[t],
                ["original_image", "heat_map"],
                ["all", "positive"],
                titles=["Original", "Noise tunnel (IG)"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False)
        fig_dias, ax_dias = viz.visualize_image_attr_multiple(
                attr_noise_ig_diastole_images[t],
                slice_images[t],
                ["original_image", "heat_map"],
                ["all", "positive"],
                titles=["Original", "Noise tunnel (IG)"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False)

        # Save plots
        fig_sys.savefig(os.path.join(slice_noise_ig_sys_out_path, f"noise_ig_systole_step{t:02}_{err_systole:.2f}err.png"))
        fig_dias.savefig(os.path.join(slice_noise_ig_dias_out_path, f"noise_ig_diastole_step{t:02}_{err_diastole:.2f}err.png"))

    if make_gifs:
        # Add the paths to the plots folders for making the animations
        gifs_folders.append(slice_ig_sys_out_path)
        gifs_folders.append(slice_ig_dias_out_path)
        gifs_folders.append(slice_noise_ig_sys_out_path)
        gifs_folders.append(slice_noise_ig_dias_out_path)

if make_gifs:
    n_proc = mp.cpu_count() if gifs_cores == 0 else gifs_cores
    pool = mp.Pool(processes=n_proc)
    print(f"Creating the GIF animations from the plots (using {n_proc} cores)...")
    pool.map(create_gif_from_folder, gifs_folders)
    print("GIF animations created!")
