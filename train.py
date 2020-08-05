import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.data_generators import Cardiac_dataset
from lib.utils import *
from models.TimeAsDepth import TimeAsDepth_0, TimeAsDepth_1, TimeAsDepth_2, TimeAsDepth_3, TimeAsDepth_4
from models.WideResNet import WideResNet50_0
from models.VGG import VGG19
from models.DenseNet import DenseNet121_0

#######################
# Training parameters #
#######################

# Parse script aguments
arg_parser = argparse.ArgumentParser(description="Runs the training of the deep learning model")
arg_parser.add_argument("target_label", help="Value to train for", type=str, choices=["Systole", "Diastole"])
arg_parser.add_argument("-e", "--epochs", help="Number of epochs to train the model", type=int, default=100)
arg_parser.add_argument("-bs", "--batch_size", help="Samples per training batch", type=int, default=128)
arg_parser.add_argument("-w", "--workers", help="Number of workers for data loading", type=int, default=2)
arg_parser.add_argument("--gpu", help="Select the GPU to use by slot id", type=int, metavar="GPU_SLOT", default=0)
arg_parser.add_argument("--multi_gpu", help="Use all the available GPU's for training", action="store_true", default=False)
arg_parser.add_argument("--pin_mem", help="To use pinned memory for data loading into GPU", type=bool, default=True)
arg_parser.add_argument("--tensorboard", help="To enable tensorboard logs", type=bool, default=True)
arg_parser.add_argument("-m", "--model", help="Select the model to train", type=str, 
        choices=["TimeAsDepth_0", "TimeAsDepth_1", "TimeAsDepth_2", "TimeAsDepth_3", "TimeAsDepth_4", "WideResNet50_0", "VGG19", "DenseNet121_0"], default="TimeAsDepth_0")
arg_parser.add_argument("-opt", "--optimizer", help="Select the training optimizer", type=str, choices=["Adam", "SGD"], default="Adam")
arg_parser.add_argument("-lr", "--learning_rate", help="Starting learning rate for the optimizer", type=float, default=0.001)
arg_parser.add_argument("-loss", "--loss_function", help="Loss function to optimize during training", type=str, choices=["MSE", "MAE"], default="MSE")
arg_parser.add_argument("-da", "--data_augmentation", help="Enable data augmentation", choices=[0, 1, 2, 3], type=int, default=0)
arg_parser.add_argument("-dp", "--data_path", help="Path to the preprocessed dataset folder", type=str, default="../preproc1_150x150_bySlices_dataset_full/")
arg_parser.add_argument("-fr", "--freeze_ratio", help="Percentaje (range [0...1]) of epochs to freeze the model from the begining", type=float, default=0.3)
arg_parser.add_argument("--use_pretrained", help="To use or not the pretrained weights if the selected model can be pretrained", type=bool, default=True)
args = arg_parser.parse_args()

data_path = args.data_path
dataset_name = get_dataset_name(data_path)
epochs = args.epochs
freeze_ratio = args.freeze_ratio
batch_size = args.batch_size
num_workers = args.workers
selected_gpu = args.gpu
multi_gpu = args.multi_gpu
pin_memory = args.pin_mem
tensorboard = args.tensorboard
target_label = args.target_label
model_name = args.model
loss_function = args.loss_function
opt_name = args.optimizer
learning_rate = args.learning_rate
data_augmentation = args.data_augmentation
use_pretrained = args.use_pretrained
exp_name = f"{dataset_name}_{target_label}_{model_name}_{opt_name}-{learning_rate}_{loss_function}"  # Experiment name
if data_augmentation > 0:
    exp_name += "_DA"
    if data_augmentation > 1: exp_name += f"{data_augmentation}"
if not use_pretrained:
    exp_name += "_no-pretrained"
print(f"Running experiment {exp_name}")

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
        print(f"Going to train with the GPU in the slot {selected_gpu} -> device model: {torch.cuda.get_device_name(selected_gpu)}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"Cuda is not available, using {device} instead")

###################
# Data generators #
###################

# Load dataset info
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
dev_df = pd.read_csv(os.path.join(data_path, "validate.csv"))
# Create train datagen
train_dataset = Cardiac_dataset(train_df, target_label, data_augmentation=data_augmentation)
train_datagen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
# Create develoment datagen
dev_dataset = Cardiac_dataset(dev_df, target_label)
dev_datagen = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

########################
# Model initialization #
########################

# Build the model
is_pretrained = False  # To unfreeze the weights
if model_name == "TimeAsDepth_0":
    model = TimeAsDepth_0()
elif model_name == "TimeAsDepth_1":
    model = TimeAsDepth_1()
elif model_name == "TimeAsDepth_2":
    model = TimeAsDepth_2()
elif model_name == "TimeAsDepth_3":
    model = TimeAsDepth_3()
elif model_name == "TimeAsDepth_4":
    model = TimeAsDepth_4()
elif model_name == "WideResNet50_0":
    model = WideResNet50_0(use_pretrained)
    is_pretrained = use_pretrained
elif model_name == "VGG19":
    model = VGG19(use_pretrained)
    is_pretrained = use_pretrained
elif model_name == "DenseNet121_0":
    model = DenseNet121_0(use_pretrained)
    is_pretrained = use_pretrained
else:
    print(f"The model name provided ({model_name}) is not valid")

# Print model architecture
print(f"Model architecture:\n {model} \n")

##################
# Training phase #
##################

# Check if we have to freeze the weights
if is_pretrained:
    model.set_freeze(True)  # Freeze pretrained weights

# Get loss function
if loss_function == "MSE":
    criterion = nn.MSELoss()
elif loss_function == "MAE":
    criterion = nn.L1Loss()
else:
    print(f"Loss function {loss_function} is not valid!")
    sys.exit()

# Get optimizer 
if opt_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif opt_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    print(f"Optimizer {opt_name} not recognized!")
    sys.exit()

# Initialization of the variables to store the results
best_loss = 99999999
best_diff = 99999999
best_epoch = -1
train_losses, test_losses = [], []
train_diffs, test_diffs = [], []

# Scheduler for changing the value of the laearning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

# Set the tensorboard writer
if tensorboard:
    tboard_writer = SummaryWriter(comment=exp_name)

# Prepare multi-gpu training if enabled
if multi_gpu and n_gpus > 1 :
    print("Preparing multi-gpu training...")
    model = nn.DataParallel(model)

# Move the model to the computing devices
model = model.to(device)

# Compute freezed epochs
freezed_epochs = int(freeze_ratio * epochs)

# Print training header
print("\n############################\n"\
      +f"# TRAIN PHASE: {epochs:>4} epochs #\n"\
      + "############################\n")

# Start training
for epoch in range(epochs):
    # Header of the epoch log
    stdout.write(f"Epoch {epoch}: ") 
    if best_epoch > -1 : stdout.write(f"current best loss = {best_loss:.5f}, at epoch {best_epoch}\n")
    else: stdout.write("\n")

    if is_pretrained and epoch == freezed_epochs:
        print("Going to unfreeze the pretrained weights")
        if multi_gpu and n_gpus > 1 :
            model.module.set_freeze(False)
        else:
            model.set_freeze(False)

    # Train split
    train_loss, train_diff = train_regresor(train_datagen, model, criterion, optimizer, device, pin_memory)
    # Development split 
    test_loss, test_diff = test_regresor(dev_datagen, model, criterion, device, pin_memory)
    # Apply the lr scheduler
    scheduler.step(test_loss)
    # Save the results of the epoch
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_diffs.append(train_diff)
    test_diffs.append(test_diff)
    # Log in tensorboard 
    if tensorboard:
        # Loss
        tboard_writer.add_scalar("Loss/train", train_loss, epoch)
        tboard_writer.add_scalar("Loss/test", test_loss, epoch)
        tboard_writer.add_scalar("Diff/train", train_diff, epoch)
        tboard_writer.add_scalar("Diff/test", test_diff, epoch)

    # If val_loss improves we store the model
    if test_losses[-1] < best_loss:
        model_path = f"models/checkpoints/{exp_name}_best"
        print(f"Saving new best model in {model_path}")
        # Save the entire model
        torch.save(model, model_path)
        # Update best model stats
        best_loss = test_losses[-1]
        best_diff = test_diffs[-1]
        best_epoch = epoch

    # To separate the epochs outputs  
    stdout.write("\n")

if tensorboard:
    tboard_writer.add_hparams(
            {"dataset": dataset_name,
            "label": target_label,
            "model": model_name,
            "pretrained": is_pretrained,
            "DA": data_augmentation,
            "optimizer": opt_name,
            "loss_func": loss_function,
            "lr": learning_rate},
            {"hparam/loss": best_loss, 
            "hparam/diff": best_diff, 
            "hparam/best_epoch": best_epoch})

# Close the tensorboard writer
if tensorboard:
    tboard_writer.close()

# Plot loss and accuracy of training epochs
plot_results(train_losses, test_losses, title=f"Loss {target_label}", save_as=exp_name + "_loss")
plot_results(train_diffs, test_diffs, title=f"Diff {target_label}", save_as=exp_name + "_diff")
