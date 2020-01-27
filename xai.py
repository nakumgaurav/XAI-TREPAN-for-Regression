import torch
import torch.nn as nn
import torch.nn.functional as torchfunc


class LinearNetwork(nn.Module):

    def __init__(self, layers_size, final_layer_function, activation_function, bias=False):
        
        self.final_layer_function = final_layer_function
        self.activation_function = activation_function

        self.bias = bias

        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1], bias=bias)
                                            for i in range(len(layers_size) - 1)])

    def forward(self, x):
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation_function(x)

        x = self.linear_layers[-1](x)
        return self.final_layer_function(x)



import csv
import os
import random

# from tensorboardX import SummaryWriter
from torch.optim import Adam

import numpy as np

# modeldir = "./model/"
# os.makedirs(modeldir, exist_ok=True)

# logdir = "./tensorboard_log/"
# TODO make logfile name based on arguments
# logfile = "logfile"

# writer = SummaryWriter(logdir + logfile)

torch.manual_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--val_size", default=0.1)
parser.add_argument("--test_size", default=0.1)
parser.add_argument("--bs", default=512)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--num_epochs", default=100)

# dataset_file = 'combined.csv'
dataset_file = args.data_path

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

dev_fraction = args.val_size
test_fraction = args.test_size
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

training_batch_size = args.bs
learning_rate = args.lr
num_epochs = args.num_epochs
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




from scipy.stats import gaussian_kde
import collections
import copy

class Oracle:
    def __init__(self, dataset):
#         print(dataset[0][0])
        self.X = np.array([tup[0] for tup in dataset])
        self.y = np.array([tup[1] for tup in dataset])
        self.num_features = self.X.shape[-1]
        self.construct_training_distribution()

    def construct_training_distribution(self):
        """Get the density estimates for each feature using a kernel density estimator.
        Any estimator could be used as described here:
        https://ned.ipac.caltech.edu/level5/March02/Silverman/paper.pdf """
        self.train_dist = []

        for i in range(self.num_features):
            data = self.X[:,i]
            kernel = gaussian_kde(data, bw_method='silverman')
            self.train_dist.append(kernel)

    def generate_instance(self, constraint):
        """Given the constraints that an instance must satisfy, generate an instance """
        instance = np.zeros(self.num_features)
        for feature_no in self.num_features:
            sampled_val = None
            while True:
                sampled_val = self.train_dist[feature_no].sample(random_state=0)
                if constraint.satisfy_feature(sampled_val, feature_no):
                    break

            instance[feature_no] = sampled_val

        return instance

    def generate_instances(self, constraint, size):
        print("Need to generate %d instances"%size)
        new_instances = []
        for i in range(size):
            new_instances.append(self.generate_instance(constraint))

        return np.array(new_instances)


    def get_regression_values(self, samples):
        """Query the network to get the regression value of the samples """
        samples = torch.Tensor(samples)
        pred = linear_regressor(samples)
        return pred.detach().numpy()


class Constraint:
    def __init__(self):
        self.constraint = []

    def add_rule(self, rule):
        self.constraint.append(rule)

    def satisfy_feature(self, value, feat_no):
        """ Given a feature value, check whether feat_no feature satisfies the constraint """
        ans = True
        for rule in self.constraint:
            feature_no, symbol, thresh = rule.split(" ")
            feature_no = int(feature_no[1:])
            if feature_no != feat_no:
                continue
            
            thresh = float(thresh)
            if symbol == "<=":
                ans &= value <= thresh
            elif symbol == ">":
                ans &= value > thresh

        return ans

    def satisfy(self, instance):
        """Given an instance, check whether it satisfies the constraint """
        ans = True
        for feat_no in range(len(instance)):
            ans &= self.satisfy_feature(instance[feat_no], feat_no)

        return ans

    def get_constrained_features(self):
            """ Gives the list of indices for features on which constraint rules are present """
            feature_indices = []
            for rule in self.constraint:
                feature_no, _, _ = rule.split(" ")
                feature_no = int(feature_no[1:])
                feature_indices.append(feature_no)
            return list(set(feature_indices))


class Node():  
  def __init__(self,data,constraints,reg_val):
    self.data = data
    self.constraints = constraints
    self.reg_val = reg_val
    self.num_features = data.shape[1]
    self.left = None
    self.right = None
    
    self.blacklisted_features = self.constraints.get_constrained_features()

class Tree:
    def __init__(self, oracle):
        self.oracle = oracle
        self.initial_data = oracle.X
        self.num_examples = len(oracle.X)
        self.tree_params = {"tree_size": 15, "split_min": 200, "l1e_threshold": 0.01, "num_feature_splits": 10}
        self.num_nodes = 0
        self.max_levels = 0

    def get_priority(self, node):
        reach_n = float(len(node.data))/ self.num_examples
        # print(f"reach_n={reach_n}")
        fidelity_n = self.get_fidelity(node)
        # print(f"fidelity_n={fidelity_n}")
        priority = -1 * (reach_n) * (1 - fidelity_n)
        return priority
        
    def get_fidelity(self, node):
#         num_corr_predictions = 0
        predictions = []
        for instance in node.data:
            prediction = self.predict(instance, self.root)
            predictions.append(prediction)
#             print(prediction, self.oracle.get_regression_values(instance.reshape(1,-1))[0])
#             print(self.oracle.get_regression_values(instance)[0], self.oracle.get_regression_values(instance).shape)
        predictions = np.array(predictions)
        l2e = np.exp(-np.sum((prediction - self.oracle.get_regression_values(node.data).squeeze())**4))
#             num_corr_predictions += 1 if l1e <= self.tree_params["l1e_threshold"] else 0

#         fidelity = float(num_corr_predictions)/float(len(node.data))
        
        return l2e

    def build_tree(self):
        """Main method which builds the tree and returns the root 
        through which the enitre tree can be accessed"""
        import queue as Q
        node_queue = Q.PriorityQueue(maxsize=self.tree_params["tree_size"])

        self.root = self.construct_node(self.initial_data, Constraint())
        self.num_nodes += 1
        node_queue.put((self.get_priority(self.root), self.root), block=False)
        
        while not node_queue.empty() and self.num_nodes <= self.tree_params["tree_size"]:
            print("num_nodes = ", self.num_nodes)
            priority, node = node_queue.get()
            node = self.add_instances(node)
            node, left_node, right_node = self.split(node)

            left_prio = self.get_priority(node.left)
            right_prio = self.get_priority(node.right)
            # print("left_prio=", left_prio)
            # print("right_prio=", right_prio)
            
            print("Node Rule : ", node.split_rule)
            node_queue.put((left_prio, left_node), block=False)
            print("# of samples in left child = ", len(left_node.data))
            node_queue.put((right_prio, right_node), block=False)
            print("# of samples in right child = ", len(right_node.data))
            self.num_nodes += 2
        
        return self.root

    def add_instances(self, node):
        """Query the oracle to add more instances to the node if required """
        num_instances = len(node.data)
        print("num_instances here=", num_instances)
        s_min = self.tree_params["split_min"]
        if num_instances >= s_min:
            return node
        
        num_new_instances = s_min-num_instances
        new_instances = self.oracle.generate_instances(node.constraints, size=num_new_instances)
        
        new_data = np.zeros(shape=(s_min, self.oracle.num_features))
        new_data[:num_instances] = node.data
        new_data[num_instances:] = new_instances
        node.data = new_data

        return node
        

    def get_best_split(self, node):
        min_mse = float("inf")
        best_split_point = None
        best_feat = None

        for i in range(self.oracle.num_features):
            # if i in node.blacklisted_features:
                # continue

            split_point, mse = self.feature_split(node.data[:,i])

            if mse < min_mse:
                best_feat = i
                best_split_point = split_point
                min_mse = mse

        return best_feat, best_split_point

    def split(self, node):
        """Decide the best split and split the node """
        best_feat, best_split_point = self.get_best_split(node)

        left_ind = node.data[:, best_feat] <= best_split_point
        right_ind = node.data[:, best_feat] > best_split_point

        left_constraints = copy.deepcopy(node.constraints)
        right_constraints = copy.deepcopy(node.constraints)
        left_rule = f"x{best_feat} <= {best_split_point}"
        right_rule = f"x{best_feat} > {best_split_point}"
        
        left_constraints.add_rule(left_rule)
        right_constraints.add_rule(right_rule)

        left_node = self.construct_node(node.data[left_ind], left_constraints)
        right_node = self.construct_node(node.data[right_ind], right_constraints)
        
        node.left = left_node#.deepcopy()
        node.right = right_node#.deepcopy()

        node.split_rule = left_rule

        return node, left_node, right_node

    def calc_mse(self, data):
        mean = np.mean(data)
        return np.mean((data-mean)**2)

    def feature_split(self, feature_data):
        split_points = np.linspace(start=0, stop=1, num=self.tree_params["num_feature_splits"])[1:-1]
        min_mse = float("inf")
        best_split_point = None

        for split_point in split_points:
            data_left = feature_data[feature_data <= split_point]
            data_right = feature_data[feature_data > split_point]
            mse = self.calc_mse(data_left) + self.calc_mse(data_right)

            if mse < min_mse:
                best_split_point = split_point
                min_mse = mse

        return best_split_point, mse
            

    def is_leaf(self, node):
        return node.left is None and node.right is None

    def predict(self, instance, root):
        if self.is_leaf(root):
            return root.reg_val
        if root.constraints.satisfy(instance):
            return self.predict(instance, root.left)
        else:
            return self.predict(instance, root.right)


    def construct_node(self, data, constraints):
        """ Input Args - data: the training data that this node has
            Output Args - A Node variable
        """
        reg_val = np.mean(self.oracle.get_regression_values(data))
        return Node(data, constraints, reg_val)

    # print tree
    
    
    
    def assign_levels(self,root,level):
      root.level = level
      if level > self.max_levels:
        self.max_levels = level
        
      if root.left is not None:
        self.assign_levels(root.left,level+1)
      if root.right is not None:
        self.assign_levels(root.right,level+1)
    
    def print_tree_levels(self,root):
      for level in range(self.max_levels+1):
        self.print_tree(root,level)
      
    def print_tree(self, root,level):
      if self.is_leaf(root):
        if level == root.level:
          print(root.reg_val," ")
      else:
        if level == root.level:
          print(root.split_rule)

        if root.left is not None:
          self.print_tree(root.left,level)
        if root.right is not None:
          self.print_tree(root.right,level)

oracle = Oracle(training_data)
print("oracle object built.")
tree_obj = Tree(oracle)
print("Now building tree...")
root = tree_obj.build_tree()
tree_obj.assign_levels(root,0)
tree_obj.print_tree_levels(root)          

test_data_torch = np.array([tup[0] for tup in test_data])
test_data_tree = test_data_torch[:3000]
test_data_torch = test_data_torch[:3000]
# print(type(test_data_torch))
test_data_torch = torch.Tensor(test_data_torch)
predi_torch = linear_regressor(test_data_torch).detach().numpy().squeeze()

predi_tree = []
for instance in test_data_tree:
    predi_tree.append(tree_obj.predict(instance, root))
    
predi_tree = np.array(predi_tree)

print(predi_torch.shape, predi_tree.shape)


def fidelity(l1e_thresh=0.01, l2e_thresh=0.0001):
    l1e = np.abs(predi_torch - predi_tree)
    # l2e = (predi_torch - predi_tree)**2
    mse = np.sum((predi_torch - predi_tree)**4)

    fidelity1 = np.exp(-mse)
    fidelity2 = np.sum(l1e < l1e_thresh) / np.size(predi_torch)
    # fidelity3 = np.sum(l2e < l2e_thresh) / np.size(predi_torch)
    # print("Fidelity = ", fidelity1*100, "%")
    # print("Fidelity = ", fidelity2*100, "%")
    # print("Fidelity = ", fidelity3*100, "%")
  
fidelity()
