import csv
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
# from tensorboardX import SummaryWriter
from torch.optim import Adam

import numpy as np

from network import LinearNetwork

# modeldir = "./model/"
# os.makedirs(modeldir, exist_ok=True)

# logdir = "./tensorboard_log/"
# TODO make logfile name based on arguments
# logfile = "logfile"

# writer = SummaryWriter(logdir + logfile)

torch.manual_seed(0)

dataset_file = 'combined.csv'

with open(dataset_file) as csvfile:
    csvreader = csv.DictReader(csvfile)

    # Extracting feature names from csv
    headers = csvreader.fieldnames
    print("Headers ", len(headers), headers)

    features = headers[1:-3]
    output = headers[-1:]
    feature_size = len(features)
    output_size = len(output)

    print("Features ", len(features), features)
    print("Output ", output)

    # Creating dataset
    dataset = []
    for line in csvreader:
        x = []
        for feature in features:
            x.append(float((line[feature])))
        y = float(line[output[0]])
        dataset.append((x, y))

    print("Complete dataset", len(dataset), dataset[0])

dev_fraction = 0.1
test_fraction = 0.1
train_fraction = 1 - dev_fraction - test_fraction

training_data = []
dev_data = []
test_data = []

dataset_size = len(dataset)
training_data_size = int(dataset_size * train_fraction)
dev_data_size = int(dataset_size * dev_fraction)
test_data_size = dataset_size - training_data_size - dev_data_size

# Preprocessing of data
# Normalise data for each feature

min_feature_values = [10000000000] * feature_size
max_feature_values = [-10000000000] * feature_size

# Find min and max value for each feature
for data in dataset:
    for feature_index in range(feature_size):
        if data[0][feature_index] < min_feature_values[feature_index]:
            min_feature_values[feature_index] = data[0][feature_index]
        if data[0][feature_index] > max_feature_values[feature_index]:
            max_feature_values[feature_index] = data[0][feature_index]

normalised_dataset = []
for data in dataset:
    x = data[0]
    normalised_x = []
    for i in range(feature_size):
        normalised_x.append((x[i] - min_feature_values[i]) / (max_feature_values[i] - min_feature_values[i]))
    normalised_dataset.append((normalised_x, data[1]))

training_data = normalised_dataset[0:training_data_size]
dev_data = normalised_dataset[training_data_size: training_data_size + dev_data_size]
test_data = normalised_dataset[-test_data_size:]

print(normalised_dataset[0])

print("Training data size ", training_data_size)
print("Dev data size ", dev_data_size)
print("Test data size ", test_data_size)

# Network definition

layers_size = [feature_size, 100,100,100,100, output_size]

linear_regressor = LinearNetwork(layers_size=layers_size, final_layer_function=torchfunc.relu,
                                 activation_function=torchfunc.relu, bias=True)

print(linear_regressor)

# Training

global_step = 1
best_model_loss = np.inf

training_batch_size = 512
learning_rate = 0.0001
num_epochs = 1000
save_period = 5

optimizer = Adam(params=linear_regressor.parameters(), lr=learning_rate)
mseloss = nn.MSELoss()

def xavier_initialisation(*module):
    modules = [*module]
    for i in range(len(modules)):
        if type(modules[i]) in [nn.Linear]:
            nn.init.xavier_normal_(modules[i].weight.data)

xavier_initialisation(linear_regressor)

for epoch in range(num_epochs):
    epoch_loss = 0
    shuffled_training_data = random.shuffle(training_data)

    num_batches = int(training_data_size / training_batch_size)

    for batch_id in range(num_batches):
        # Get minibatch from training data
        data = []
        outputs = []

        for i in range(batch_id * training_batch_size, (batch_id + 1) * training_batch_size):
            data.append(training_data[i][0])
            outputs.append(training_data[i][1])
        data = torch.Tensor(data)
        outputs = torch.Tensor(outputs)

        predicted_values = linear_regressor(data).squeeze()
        # print(predicted_values[0:10],outputs[0:10])
        batch_loss = mseloss(predicted_values, outputs)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss
        # writer.add_scalar("Batch loss", batch_loss, global_step)
        global_step += 1
        # print(f"Batch completed. Loss = {batch_loss}")

    # writer.add_scalar("Epoch loss", epoch_loss, epoch)
    print(f"Epoch {epoch} completed. Loss = {epoch_loss}")
    # break

    # if epoch_loss < best_model_loss:
    #     print(f"Found new best model")
    #     best_model_loss = epoch_loss
        # torch.save(linear_regressor.state_dict(), modeldir + "best_model.pt")

    # if epoch % save_period == 0:
        # print(f"Saving periodically for epoch {epoch}")
        # torch.save(linear_regressor.state_dict(), modeldir + "periodic_model.pt")
