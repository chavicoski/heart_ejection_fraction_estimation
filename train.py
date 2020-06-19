import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.data_generators import Cardiac_dataset
from lib.utils import *
from models.my_models import Time_as_depth_model

# Check computing device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"{n_gpus} GPU's available:")
        for gpu_idx in range(n_gpus):
            print(f"\t-At device cuda:{gpu_idx} -> device model = {torch.cuda.get_device_name(gpu_idx)}")
    else:
        print(f"Cuda available with device {device} -> device model = {torch.cuda.get_device_name(device_slot)}")
    
    # Select a GPU
    device_slot = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_slot}")
else:
    n_gpus = 0
    device = torch.device("cpu")
    print(f"Cuda is not available, using {device} instead")
 

#######################
# Training parameters #
#######################

epochs = 100
batch_size = 32
num_workers = 2   # Processes for loading data in parallel
multi_gpu = True  # Enables multi-gpu training if it is possible
pin_memory = True  # Pin memory for extra speed loading batches in GPU
tensorboard = True  # Enable tensorboard
target_label = "Systole"  # Target label to predict
model_name = "TimeAsDepth"  # Model architecture name
opt_name = "Adam"  # Selected optimizer
learning_rate = 0.01  # Learning rate for the optimizer
momentum = 0.9  # In case of opt_name="SGD"
exp_name = f"{target_label}_{model_name}_{opt_name}-{learning_rate}"  # Experiment name
print(f"Running experiment {exp_name}")

###################
# Data generators #
###################

# Load dataset info
train_df = pd.read_csv("../preproc1_150x150_bySlices_dataset/train.csv")
dev_df = pd.read_csv("../preproc1_150x150_bySlices_dataset/validate.csv")
# Create train datagen
train_dataset = Cardiac_dataset(train_df, target_label)
train_datagen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
# Create develoment datagen
dev_dataset = Cardiac_dataset(dev_df, target_label)
dev_datagen = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

########################
# Model initialization #
########################

# Build the model
model = Time_as_depth_model()
# Print model architecture
print(f"Model architecture:\n {model} \n")

##################
# Training phase #
##################

# Get loss function
criterion = nn.MSELoss()
# Get optimizer 
if opt_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif opt_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
else:
    print(f"Optimizer {opt_name} not recognized!")
    sys.exit()

# Initialization of the variables to store the results
best_loss = 99999
best_epoch = -1
train_losses, test_losses = [], []

# Scheduler for changing the value of the laearning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

# Set the tensorboard writer
if tensorboard:
    tboard_writer = SummaryWriter(comment=exp_name)

# Prepare multi-gpu training if enabled
if multi_gpu and n_gpus > 1 :
    print("Preparing multi-gpu training...")
    model = nn.DataParallel(model)

# Move the model to the computing devices
model = model.to(device)

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
    # Train split
    train_loss = train_regresor(train_datagen, model, criterion, optimizer, device, pin_memory)
    # Development split 
    test_loss = test_regresor(dev_datagen, model, criterion, device, pin_memory)
    # Apply the lr scheduler
    scheduler.step(test_loss)
    # Save the results of the epoch
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    # Log in tensorboard 
    if tensorboard:
        # Loss
        tboard_writer.add_scalar("Loss/train", train_loss, epoch)
        tboard_writer.add_scalar("Loss/test", test_loss, epoch)

    # If val_loss improves we store the model
    if test_losses[-1] < best_loss:
        model_path = f"models/checkpoints/{exp_name}_best"
        print(f"Saving new best model in {model_path}")
        # Save the entire model
        torch.save(model, model_path)
        # Update best model stats
        best_loss = test_losses[-1]
        best_epoch = epoch

    # To separate the epochs outputs  
    stdout.write("\n")

if tensorboard:
    tboard_writer.add_hparams(
        {"label": target_label,
        "model": model_name,
        "optimizer": opt_name,
        "lr": learning_rate},
        {"hparam/loss": best_loss, 
        "hparam/best_epoch": best_epoch})

# Close the tensorboard writer
if tensorboard:
    tboard_writer.close()

# Plot loss and accuracy of training epochs
plot_results_regresor(train_losses, test_losses, title=f"Loss {target_label}",save_as=exp_name)
