import os
from sys import stdout
from time import time
import torch
from matplotlib import pyplot as plt

def get_dataset_name(path):
    """Given a path to a folder this function returns tha name of this folder"""
    parts = os.path.split(path)
    if parts[-1] is not '':  # To fix the case ending with '/'
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
        running_diff += torch.mean(torch.abs(outputs - target))
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
            test_diff += torch.mean(torch.abs(output - target))

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
    if save_as is not "": 
        plt.savefig("plots/train_results/" + save_as + ".png")
    plt.show() 
