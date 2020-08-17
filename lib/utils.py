import os
import numpy as np
import sys
from sys import stdout
from time import time
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
from array2gif import write_gif
from scipy.stats import norm

def create_gif_from_folder(path):
    '''Given a path to a folder with png images, this function takes all the images
    and creates a gif from them inside the folder'''
    images = []
    for f_name in sorted(os.listdir(path)):
        if f_name.endswith(".png"):
            images.append(cv2.imread(os.path.join(path, f_name)))
    
    write_gif(np.array(images), os.path.join(path, "animation.gif"), fps=30)

def to_RGB_images(x):
    '''Given a pytorch tensor with shape (1, timesteps, H, W) this functions returns
    a list of length 'timesteps' with RGB images of shape (H, W, 3)
    '''
    images = []
    # From (1, timesteps, H, W) to (H, W, timesteps)
    x_trans = np.transpose(x.squeeze().cpu().detach().numpy(), (1,2,0))
    for t in range(x.size(1)):
        # Create the 3 channel image and store it contiguous
        images.append(np.dstack([x_trans[:,:,t,None]] * 3).copy(order="C"))

    return images


def get_dataset_name(path):
    '''Given a path to a folder this function returns tha name of this folder'''
    parts = os.path.split(path)
    if parts[-1] != '':  # To fix the case ending with '/'
        return parts[-1]
    else:
        return os.path.basename(parts[-2])


def train_regresor(train_loader, net, criterion, optimizer, device, pin_memory):
    '''
    Training loop function for a regresion model
    Params:
        train_loader -> pytorch DataLoader for training data
        net -> pytorch model
        criterion -> pytorch loss function
        optimizer -> pytorch optimizer
        device -> pytorch computing device
    '''
    # Set the net in train mode
    net.train()

    # Initialize stats
    running_loss = 0.0
    running_diff = 0.0
    samples_count = 0

    # Epoch timer
    epoch_timer = time()

    # Training loop
    for batch_idx, batch in enumerate(train_loader, 0):
        # Batch timer
        batch_timer = time()
        # Get input and target from batch
        data, target = batch["X"], batch["Y"]
        # Move tensors to computing device
        data = data.to(device, non_blocking=pin_memory)
        target = target.to(device, non_blocking=pin_memory)
        # Accumulate the number of samples in the batch 
        samples_count += len(data)
        # Reset gradients
        optimizer.zero_grad()
        # Forward
        outputs = net(data)
        loss = criterion(outputs, target)
        # Backward
        loss.backward()
        optimizer.step()
        # Accumulate loss
        running_loss += loss.item()
        # Compute diff
        running_diff += torch.abs(outputs - target).sum()
        # Compute current statistics
        current_loss = running_loss / samples_count 
        current_diff = running_diff / samples_count 
        # Compute time per batch (in miliseconds)
        batch_time = (time() - batch_timer) * 1000
        # Print training log
        stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - diff: {current_diff:.2f}ml")

    # Compute total epoch time (in seconds)
    epoch_time = time() - epoch_timer
    # Final print with the total time of the epoch
    stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {epoch_time:.1f}s {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - diff: {current_diff:.2f}ml")
    # Return final loss value
    return current_loss, current_diff


def test_regresor(test_loader, net, criterion, device, pin_memory):
    '''
    Test function. Computes loss and accuracy for development set. For a 
    regresion model
    Params:
        test_loader -> pytorch DataLoader for testing data
        net -> pytorch model
        criterion -> pytorch loss function
        device -> pytorch computing device
    '''
    # Set the net in eval mode
    net.eval()

    # Initialize stats
    test_loss = 0.0 
    test_diff = 0.0

    # Test timer
    test_timer = time()

    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        for batch in test_loader:       
            # Get input and target from batch
            data, target = batch["X"], batch["Y"]
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)       
            # Compute forward and get output logits
            output = net(data)       
            # Compute loss and accumulate it
            test_loss += criterion(output, target).item()       
            # Compute diff
            test_diff += torch.abs(output - target).sum()

    # Compute final loss
    test_loss /= len(test_loader.dataset)   
    # Compute final diff
    test_diff /= len(test_loader.dataset)   
    # Compute time consumed 
    test_time = time() - test_timer
    # Print test log 
    stdout.write(f'\nTest {test_time:.1f}s: val_loss: {test_loss:.5f} - diff: {test_diff:.2f}ml\n')    
    # Return final loss value
    return test_loss, test_diff


def get_PDF(systole_value, diastole_value, mode="cdf", mae=[10, 20]):
    '''
    Given the predicted volume values for systole and diastole computes the
    PDF for the submission file of the competition.
    Params:
        systole_value -> predicted volume of systole
        diastole_value -> predicted volume of diastole
        mode -> choose from ["cdf", "step"] the way to compute the PDF
        mae -> mean squared error for systole and diastole during validation
    '''
    if mode == "cdf":
        systole_probs = list(norm.cdf(range(600), int(systole_value), mae[0]))
        diastole_probs = list(norm.cdf(range(600), int(diastole_value), mae[1]))
    elif mode == "step": 
        systole_probs = [0 if i < int(systole_value) else 1 for i in range(600)]
        diastole_probs = [0 if i < int(diastole_value) else 1 for i in range(600)]
    else:
        print(f"lib.utils.get_PDF(): The mode {mode} is not valid!")

    return systole_probs, diastole_probs


def submission_regresor(test_loader, net_systole, net_diastole, device, pin_memory, mode="cdf", mae=[15, 20]):
    '''
    Submission function. Generates the submission CSV from the test data. For a 
    regresion model
    Params:
        test_loader -> pytorch DataLoader for testing data
        net_systole -> pytorch model for systole estimation
        net_diastole -> pytorch model for diastole estimation
        device -> pytorch computing device
        mode -> choose from ["cdf", "step"] the way to compute the PDF
        mae -> mean squared error for systole and diastole during validation

    Note: The batch_size of the test_loader must be 1
    '''
    # Set the nets in eval mode
    net_systole.eval()
    net_diastole.eval()
    
    # Initialize results dataframe
    df = pd.DataFrame(columns = ["Id"] + [f"P{i}" for i in range(600)])

    # To store the ID os the cases without data for the selected view
    no_data_cases = []

    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        df_idx = 0
        for batch in tqdm(test_loader):       
            # Get input and target from batch
            id_, data, target_systole, target_diastole = batch["ID"], batch["X"], batch["Y_systole"], batch["Y_diastole"]
            if data.size(1) == 0:  # Check if there is data
                no_data_cases.append(id_[0])
                continue
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)[0]
            # Compute forward and get output predictions
            pred_systole = net_systole(data)       
            pred_diastole = net_diastole(data)       
            # Compute mean from the predictions for each slice of the case
            systole_value = pred_systole.mean()
            diastole_value = pred_diastole.mean()
            # Compute the prob for each ml value (from 0 to 599)
            systole_probs, diastole_probs = get_PDF(systole_value, diastole_value, mode, mae)
            # Add the pred to the dataframe
            df.loc[df_idx] = [f"{int(id_[0])}_Systole"] + systole_probs
            df.loc[df_idx+1] = [f"{int(id_[0])}_Diastole"] + diastole_probs
            df_idx += 2

    return df, no_data_cases


def complete_missing_with_SAX(df, missing_cases, test_loader, net_systole, net_diastole, device, pin_memory, mode="cdf", mae=[15, 20]):
    '''
    Completes the given DataFrame with the cases that are not predicted.
    Params:
        df -> DataFrame to complete
        missing_cases -> A list with the cases without estimation
        test_loader -> pytorch DataLoader for testing data
        net_systole -> pytorch model for systole estimation
        net_diastole -> pytorch model for diastole estimation
        device -> pytorch computing device
        mode -> choose from ["cdf", "step"] the way to compute the PDF
        mae -> mean squared error for systole and diastole during validation

    Note: The batch_size of the test_loader must be 1
    '''
    # Set the nets in eval mode
    net_systole.eval()
    net_diastole.eval()
    
    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        df_idx = len(df.index)  # To append at the end
        for batch in tqdm(test_loader):       
            # Get input and target from batch
            id_, data, target_systole, target_diastole = batch["ID"], batch["X"], batch["Y_systole"], batch["Y_diastole"]
            if id_[0] not in missing_cases: continue
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)[0]
            # Compute forward and get output predictions
            pred_systole = net_systole(data)       
            pred_diastole = net_diastole(data)       
            # Compute mean from the predictions for each slice of the case
            systole_value = pred_systole.mean()
            diastole_value = pred_diastole.mean()
            # Compute the prob for each ml value (from 0 to 599)
            systole_probs, diastole_probs = get_PDF(systole_value, diastole_value, mode, mae)
            # Add the pred to the dataframe
            df.loc[df_idx] = [f"{int(id_[0])}_Systole"] + systole_probs
            df.loc[df_idx+1] = [f"{int(id_[0])}_Diastole"] + diastole_probs
            df_idx += 2

    return df


def train_classifier(train_loader, net, criterion, optimizer, device, pin_memory):
    '''
    Training loop function for a classifier model
    Params:
        train_loader -> pytorch DataLoader for training data
        net -> pytorch model
        criterion -> pytorch loss function
        optimizer -> pytorch optimizer
        device -> pytorch computing device
    '''
    # Set the net in train mode
    net.train()

    # Initialize stats
    running_loss = 0.0
    running_correct = 0.0
    samples_count = 0

    # Epoch timer
    epoch_timer = time()

    # Training loop
    for batch_idx, batch in enumerate(train_loader, 0):
        # Batch timer
        batch_timer = time()
        # Get input and target from batch
        data, target = batch["X"], batch["Y"]
        # Move tensors to computing device
        data = data.to(device, non_blocking=pin_memory)
        target = target.to(device, non_blocking=pin_memory)
        # Accumulate the number of samples in the batch 
        samples_count += len(data)
        # Reset gradients
        optimizer.zero_grad()
        # Forward
        outputs = net(data)
        loss = criterion(outputs, target)
        # Backward
        loss.backward()
        optimizer.step()
        # Accumulate loss
        running_loss += loss.item()
        # Accumulate accuracy
        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        running_correct += correct
        # Compute current statistics
        current_loss = running_loss / samples_count 
        current_acc = running_correct / samples_count
        # Compute time per batch (in miliseconds)
        batch_time = (time() - batch_timer) * 1000
        # Print training log
        stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - acc: {current_acc:.5f}")

    # Compute total epoch time (in seconds)
    epoch_time = time() - epoch_timer
    # Final print with the total time of the epoch
    stdout.write(f"\rTrain batch {batch_idx+1}/{len(train_loader)} - {epoch_time:.1f}s {batch_time:.1f}ms/batch - loss: {current_loss:.5f} - acc: {current_acc:.5f}")
    # Return final loss and accuracy
    return current_loss, current_acc


def test_classifier(test_loader, net, criterion, device, pin_memory):
    '''
    Test function. Computes loss and accuracy for development set. For a 
    classifier model
    Params:
        test_loader -> pytorch DataLoader for testing data
        net -> pytorch model
        criterion -> pytorch loss function
        device -> pytorch computing device
    '''
    # Set the net in eval mode
    net.eval()

    # Initialize stats
    test_loss = 0   
    correct = 0

    # Test timer
    test_timer = time()

    # Set no_grad to avoid gradient computations
    with torch.no_grad():     
        # Testing loop
        for batch in test_loader:       
            # Get input and target from batch
            data, target = batch["X"], batch["Y"]
            # Move tensors to computing device
            data = data.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)       
            # Compute forward and get output logits
            output = net(data)       
            # Compute loss and accumulate it
            test_loss += criterion(output, target).item()       
            # Compute correct predictions and accumulate them
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Compute final loss
    test_loss /= len(test_loader.dataset)   
    # Compute final accuracy
    test_acc = correct / len(test_loader.dataset) 
    # Compute time consumed 
    test_time = time() - test_timer
    # Print test log 
    stdout.write(f'\nTest {test_time:.1f}s: val_loss: {test_loss:.5f} - val_acc: {test_acc:.5f}\n')    

    return test_loss, test_acc


def plot_results(train_values, test_values, title="", save_as=""):
    '''
    Given the list of metric values for each epoch during training, this 
    function shows and saves(if told) the plot of the values
    Params:
        save_as -> name of the plot file without extension. If it
                   is not specified the function don't save the plot
    '''
    plt.plot(train_values, "r", label="train")
    plt.plot(test_values, "g", label="test")
    plt.legend()
    plt.title(title)
    if save_as != "": 
        plt.savefig("plots/train_results/" + save_as + ".png")
    else:
        plt.show() 
