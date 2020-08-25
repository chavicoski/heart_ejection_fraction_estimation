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
from captum.attr import IntegratedGradients, Saliency
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
arg_parser.add_argument("--view", help="Type of view of the model", type=str, choices=["SAX", "2CH", "4CH"], default="SAX")
arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, default="../preproc1_150x150_bySlices_dataset_allViews/")
arg_parser.add_argument("--split", help="Data split to take images", type=str, choices=["train", "validate", "test"], default="validate")
arg_parser.add_argument("-out", "--output_path", help="Folder to save the plots of the attributions", type=str, default="plots/interpretability")
arg_parser.add_argument("-gifs", "--make_gifs", help="Enable the gifs creation", choices=[0, 1], type=int, default=1)
arg_parser.add_argument("-rand", "--random_sample", help="To take a random sample", choices=[0, 1], type=int, default=1)
arg_parser.add_argument("--nproc_gifs", help="Number of processes to create the gifs in parallel (0 = all cores)", choices=[i for i in range(mp.cpu_count() + 1)], type=int, default=0)
args = arg_parser.parse_args()

systole_path = args.systole_model
diastole_path = args.diastole_model
view = args.view
data_path = args.data_path
out_path = args.output_path
split = args.split
num_workers = args.workers
selected_gpu = args.gpu
pin_memory = args.pin_mem
make_gifs = bool(args.make_gifs)
random_sample = bool(args.random_sample)
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

#############################
# Get the sample to analyze #
#############################

# Load dataset info
df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
# Create test datagen
dataset = Submission_dataset(df, view=view)
datagen = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory, shuffle=random_sample)
# Load the sample to analyze
sample = next(iter(datagen))
id_, data, y_systole, y_diastole = sample["ID"], sample["X"], sample["Y_systole"], sample["Y_diastole"]
# Move to the computing device
data = data.to(device, non_blocking=pin_memory)[0]
y_systole, y_diastole = y_systole.to(device), y_diastole.to(device)
# Log
n_slices = data.size(0)
n_timesteps = data.size(1)
print(f"\nGoing to analyze case {id_.item()} from {split} split:")

##########################
# Prepare output folders #
##########################
os.makedirs(out_path, exist_ok=True)
case_path = os.path.join(out_path, f"case_{id_.item()}")
os.makedirs(case_path, exist_ok=True)
case_out_path = os.path.join(case_path, f"{view}")
os.makedirs(case_out_path, exist_ok=True)

# Create color map for the plots
default_cmap = LinearSegmentedColormap.from_list(
        'custom blue',
        [(0, '#ffffff'),
        (0.25, '#000000'),
        (1, '#000000')], N=256)

# To store the paths to the plots folders for making later the animations
if make_gifs: gifs_folders = []

for label, model_path in [("systole", systole_path), ("diastole", diastole_path)]:

    ############################
    # Select the current label #
    ############################
    if label == "systole": y = y_systole
    elif label == "diastole": y = y_diastole
    else: 
        print(f"ERROR!: The label {label} is not valid")
        sys.exit()

    ##########################
    # Load pretrained models #
    ##########################

    model = torch.load(model_path)
    # Move the model to the computing device
    model = model.to(device)
    # Set the inference mode
    model = model.eval()

    #########################
    # Intepretability phase #
    #########################

    # Get the predicted values and error for the full case classification
    pred = model(data).mean()
    err = torch.abs(pred - y).item()

    print(f"Results {label}:")
    print(f"\tGround truth: {y.item():.2f}")
    print(f"\tPredicted: {pred:.2f}")
    print(f"\tAbsolute error: {err:.2f}")

    attr_algorithms = [("Integrated Gradients", "ig"), ("Saliency", "saliency")]
    # Integrated gradients algorithm
    ig = IntegratedGradients(model)

    # Analyze every slice of the case
    print(f"Slice level analysis for {label}:")

    for slice_id in range(n_slices):
        # Get slice data with shape (1, timesteps, H, W)
        slice_data = data[slice_id].unsqueeze_(0)

        # Compute prediction and error
        pred = model(slice_data).item()
        err = pred - y.item()
        print(f"\tSlice {slice_id}: err = {err:.2f}")

        # Create the slice folder structure
        slice_out_path = os.path.join(case_out_path, f"slice_{slice_id}")
        os.makedirs(slice_out_path, exist_ok=True)
        slice_label_out_path = os.path.join(slice_out_path, f"{label}_{err:.2f}")
        os.makedirs(slice_label_out_path, exist_ok=True)

        for algorithm_name, aux_algo_name in attr_algorithms:
            # Create folder for the current algorithm
            slice_algo_out_path = os.path.join(slice_label_out_path, f"{algorithm_name}")
            os.makedirs(slice_algo_out_path, exist_ok=True)

            # Compute the algorithm attributions
            if algorithm_name == "Integrated Gradients":
                aux_algorithm = IntegratedGradients(model)
                attributions = aux_algorithm.attribute(slice_data, n_steps=400, internal_batch_size=1)
            elif algorithm_name == "Saliency":
                aux_algorithm = Saliency(model)
                attributions = aux_algorithm.attribute(slice_data, abs=False)
                attributions = torch.cat((torch.clamp(attributions, min=0), torch.abs(torch.clamp(attributions, max=0))), dim=1)
                positive_path = os.path.join(slice_algo_out_path, "positive_grads")
                negative_path = os.path.join(slice_algo_out_path, "negative_grads")
                os.makedirs(positive_path, exist_ok=True)
                os.makedirs(negative_path, exist_ok=True)
            else:
                print(f"The algorithm name {algorithm_name} is not valid!")
                sys.exit()

            # Get RGB images from the attributions
            attr_images = to_RGB_images(attributions)

            # Get RGB images from the slice for plots
            slice_images = to_RGB_images(slice_data)

            # Make plots of the selected slice and timestep
            if algorithm_name == "Saliency":
                for t in range(n_timesteps):
                    fig, ax = viz.visualize_image_attr_multiple(
                            attr_images[t],
                            slice_images[t],
                            ["original_image", "heat_map"],
                            ["all", "positive"],
                            titles=["Original", f"{algorithm_name} positive values ({label})"],
                            cmap=default_cmap,
                            show_colorbar=True,
                            use_pyplot=False)
                    # Save plots
                    fig.savefig(os.path.join(positive_path, f"positive_{aux_algo_name}_step{t:02}_{err:.2f}err.png"))

                    fig, ax = viz.visualize_image_attr_multiple(
                            attr_images[t+n_timesteps],
                            slice_images[t],
                            ["original_image", "heat_map"],
                            ["all", "positive"],
                            titles=["Original", f"{algorithm_name} negative values ({label})"],
                            cmap=default_cmap,
                            show_colorbar=True,
                            use_pyplot=False)
                    # Save plots
                    fig.savefig(os.path.join(negative_path, f"negative_{aux_algo_name}_step{t:02}_{err:.2f}err.png"))

                if make_gifs:
                    # Add the paths to the plots folders for making the animations
                    gifs_folders.append(positive_path)
                    gifs_folders.append(negative_path)
            else:
                for t in range(n_timesteps):
                    fig, ax = viz.visualize_image_attr_multiple(
                            attr_images[t],
                            slice_images[t],
                            ["original_image", "heat_map"],
                            ["all", "positive"],
                            titles=["Original", f"{algorithm_name} ({label})"],
                            cmap=default_cmap,
                            show_colorbar=True,
                            use_pyplot=False)

                    # Save plots
                    fig.savefig(os.path.join(slice_algo_out_path, f"{aux_algo_name}_step{t:02}_{err:.2f}err.png"))

                if make_gifs:
                    # Add the paths to the plots folders for making the animations
                    gifs_folders.append(slice_algo_out_path)

            # Free memory
            del attributions

        # Free memory
        del aux_algorithm
        del slice_data

    # Free memory
    del model
    del ig

if make_gifs:
    n_proc = mp.cpu_count() if gifs_cores == 0 else gifs_cores
    pool = mp.Pool(processes=n_proc)
    print(f"Creating the GIF animations from the plots (using {n_proc} cores)...")
    pool.map(create_gif_from_folder, gifs_folders)
    print("GIF animations created!")
