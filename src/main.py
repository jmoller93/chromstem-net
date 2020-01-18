#!/env/python

"""
This is the main script to train and optimize the neural network

Args:
    TBD

"""

# Import necessary libraries
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Import local classes
from dataset import ChromstemDataset
from nnet import Net

# Model trainer from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model,loaders,criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        model.train()  # Set model to training mode

        # Iterate over data.
        for i,data in enumerate(loaders['train'],0):
            inputs, labels = data['chromstem'], data['num_nucls']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # After each epoch move the scheduler forward
        scheduler.step()

        # Calculate the loss and accuracy
        epoch_loss = running_loss / len(loaders['train'])
        epoch_acc = running_corrects.double() / len(loaders['train'])

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

def main():
    # Parse the inputs
    parser = argparse.ArgumentParser(description='Inputs for the neural network trainer')
    parser.add_argument('--n_epoch','-n',type=int,help='Number of epochs to train the network with',required=True)
    args = parser.parse_args()

    # Initialize the dataset
    train_dataset = ChromstemDataset('../trains_label.csv','../')
    test_dataset  = ChromstemDataset('../tests_label.csv','../')
    val_dataset   = ChromstemDataset('../vals_label.csv','../')

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=True,num_workers=4)
    testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=4,shuffle=False,num_workers=4)
    valloader   = torch.utils.data.DataLoader(val_dataset, batch_size=4,shuffle=True,num_workers=4)
    loaders = {'train' : trainloader,
               'test'  : testloader,
               'val'   : valloader
              }

    # Initialize the neural net
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

    # Send the network to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the neural net for the appropriate number of epochs
    model = train_model(model, # Neural net
                        loaders, # Data (for now it is only training data)
                        criterion, # Loss criteria
                        optimizer, # Optimization routine
                        exp_lr_scheduler, # Learning rate scheduler
                        device, # Which device is this being run on
                        num_epochs=args.n_epoch # Number of epochs
                       )

    return

if __name__ == "__main__":
    main()
