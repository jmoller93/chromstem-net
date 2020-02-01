#!/env/python

"""
This is the main script to train and optimize the neural network

Args:
    TBD

"""

# Import necessary libraries
import os
import time
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Import local classes
from hyperparams import HyperParameters
from dataset import ChromstemDataset
from nnet import Net

# Model trainer from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, loaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    # Load best possible model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Loop through training epochs
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Loop through evaluation and training
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "val":
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_total = 0.0

            # Iterate over data.
            for i, data in enumerate(loaders[phase], 0):
                inputs, labels = data["chromstem"], data["num_nucls"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Capture running statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += labels.size(0)

            # After each training epoch move the scheduler forward
            if phase == "train":
                scheduler.step()

            # Calculate loss and accuracy for each
            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.double() / running_total * 100.0

            # Print the loss
            print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

            # Save best version of the model on validation set
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save current best version of the model
                torch.save(best_model_wts, "chromstem-net-temp.pth")

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model, best_model_wts


# Test the accuracy of the model on the testing set
def test_model(model, loader, device, batch_size):
    # Set model to evaluation mode
    model.eval()

    # Initialize accuracy calculation
    correct = 0.0
    total = 0.0
    class_correct = list(0.0 for i in range(100))
    class_total = list(0.0 for i in range(100))

    # Iterate over data.
    for i, data in enumerate(loader, 0):
        inputs, labels = data["chromstem"], data["num_nucls"]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get accuracy of each prediction as well
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        # Capture running statistics
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    # Accuracy calculation
    accuracy = correct / total * 100.0

    # Print accuracy of the model
    print("Total testing accuracy: {:.4f}".format(accuracy))
    for i in range(100):
        if class_total[i] != 0:
            print(
                "Accuracy of %3d nucls : %2.2f %%"
                % (i, 100 * class_correct[i] / class_total[i])
            )
        # else:
        # print('Accuracy of %3d nucls : 00.00 %%' % (i))


def main():
    # Parse the inputs
    parser = argparse.ArgumentParser(
        description="Inputs for the neural network trainer"
    )
    parser.add_argument(
        "--n_epoch",
        "-n",
        type=int,
        help="Number of epochs to train the network with",
        required=True,
    )
    parser.add_argument(
        "--frac",
        "-f",
        type=float,
        default=1.0,
        help="Fraction of the dataset to train on",
        required=False,
    )
    parser.add_argument(
        "--load",
        "-l",
        dest="load",
        action="store_true",
        help="Load previous model instead of training the network",
    )
    args = parser.parse_args()

    # If loading file, make sure file exists
    if args.load:
        fnme = "chromstem-net.pth"
        if not os.path.isfile(fnme):
            raise EnvironmentError(
                "File %s is not found!\n Please run this script without the -l option!\n"
                % fnme
            )
        else:
            best_wts = torch.load(fnme)

    # Make sure files exists
    labels = ["trains", "tests", "vals"]
    for label in labels:
        fnme = "../" + label + "_label.csv"
        if not os.path.isfile(fnme):
            raise EnvironmentError(
                "File %s is not found!\n Please run the label_data.sh script in the home directory!\n"
                % fnme
            )

    # Initialize the dataset
    train_dataset = ChromstemDataset("../trains_label.csv", "../", frac=args.frac)
    test_dataset = ChromstemDataset("../tests_label.csv", "../", frac=args.frac)
    val_dataset = ChromstemDataset("../vals_label.csv", "../", frac=args.frac)

    # Initialize the hyperparameters
    hyperparams = HyperParameters(batch_size=128, num_workers=8)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=hyperparams.num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=hyperparams.num_workers,
    )

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    # Initialize the neural net
    model = Net(hyperparams.filter_size, hyperparams.pool_size)

    # Send the network to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # If not loading file
    if not args.load:
        # Initialize loss function and training hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train the neural net for the appropriate number of epochs
        model, best_wts = train_model(
            model,  # Neural net
            loaders,  # Data (for now it is only training data)
            criterion,  # Loss criteria
            optimizer,  # Optimization routine
            exp_lr_scheduler,  # Learning rate scheduler
            device,  # Which device is this being run on
            num_epochs=args.n_epoch,  # Number of epochs
        )

        # Save current version of the model
        torch.save(best_wts, "chromstem-net.pth")

    # Load the best validation accuracy
    model.load_state_dict(best_wts)

    # Evaluate the models test accuracy
    test_model(model, loaders["test"], device, hyperparams.batch_size)

    return


if __name__ == "__main__":
    main()
