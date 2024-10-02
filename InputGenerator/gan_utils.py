"""
Module for all Dataset and Generator/Discriminator Classes
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import seaborn as sns
import os
import time


# Swish Activation fn (x * sigmoid(beta * x))
class Swish(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, input_tensor):
        return 2*input_tensor * torch.sigmoid(self.beta * input_tensor)


# Version with learnable parameter - ATM NOT WORKING
class LearnedSwish(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.slope * x)


class GrainDataset(torch.utils.data.Dataset):

    def __init__(self, df, label):
        super(GrainDataset, self).__init__()
        self.label = label
        # delete nans
        df.dropna(axis=1, inplace=True)
        self.result_df = df.copy()
        # print(self.result_df)
        # normalise [0,1]
        # Every Dataset normalised by own values - Correct?
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            self.result_df[feature_name] = 2 * ((df[feature_name] - min_value) / (max_value - min_value)) - 1
        # To tensor
        self.result = torch.Tensor(self.result_df.to_numpy())

    def __getitem__(self, x):
        return self.result[x], self.label

    def __len__(self):
        return len(self.result)


class Generator(nn.Module):

    def __init__(self, z_dim, num_features, depth, width, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.depth = depth
        self.input_layer = nn.Linear(z_dim, width)
        for i in range(depth):
            self.linears = nn.ModuleList([nn.Linear(width, width) for i in range(self.depth)])
        self.output_layer = nn.Linear(width, num_features)
        # print(self)

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        for i in range(self.depth):
            x = F.relu(self.linears[i](x))
        output = self.output_layer(x)
        output_scaled = torch.tanh(output)
        return output_scaled


class Discriminator(nn.Module):

    def __init__(self, p, num_features, depth, width, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.depth = depth
        self.dropout = nn.Dropout(p)
        self.input_layer = nn.Linear(num_features, width)
        for i in range(depth):
            self.linears = nn.ModuleList([nn.Linear(width, width) for i in range(self.depth)])
        self.output_layer = nn.Linear(width, 1)
        # print(self)

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        for i in range(self.depth):
            x = self.dropout(F.relu(self.linears[i](x)))
        output = self.output_layer(x)
        return output


class CGenerator(nn.Module):

    def __init__(self, z_dim, num_features, depth, width, n_classes, embed_size, activation='Relu', normalize=True):
        super(CGenerator, self).__init__()
        self.n_classes = n_classes
        self.depth = depth
        self.width = width
        self.activation = activation.lower()
        self.input_layer = nn.Linear(z_dim + embed_size, width)
        self.embed_layer = nn.Embedding(self.n_classes, embed_size)
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            self.linears.append(nn.Linear(self.width, self.width))
            if normalize:
                self.linears.append(nn.BatchNorm1d(self.width))
        self.output_layer = nn.Linear(width, num_features)
        if self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif self.activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        elif self.activation == 'swish':
            self.activation_fn = Swish(beta=1)
        else:
            self.activation_fn = nn.ReLU()

    def forward(self, x, labels):
        embed_labels = self.embed_layer(labels)
        x = torch.cat((x, embed_labels), -1)
        x = self.activation_fn(self.input_layer(x))
        for i in range(self.depth):
            x = self.activation_fn(self.linears[i](x))
        output = self.output_layer(x)
        output_scaled = torch.tanh(output)
        return output_scaled


class CDiscriminator(nn.Module):

    def __init__(self, p, num_features, depth, width, n_classes, embed_size, activation='Relu'):
        super(CDiscriminator, self).__init__()
        self.n_classes = n_classes
        self.embed_layer = nn.Embedding(self.n_classes, embed_size)
        self.embed_layer.weight.requires_grad = False
        self.depth = depth
        self.activation = activation
        self.dropout = nn.Dropout(p)
        self.input_layer = nn.Linear(num_features + embed_size, width)
        self.linears = nn.ModuleList([nn.Linear(width, width) for i in range(self.depth)])
        self.output_layer = nn.Linear(width, 1)
        if self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif self.activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        elif self.activation == 'swish':
            self.activation_fn = Swish(beta=1)
        else:
            self.activation_fn = nn.ReLU()

    def forward(self, x, labels):
        embed_labels = self.embed_layer(labels)
        x = torch.cat((x, embed_labels), -1)
        x = self.activation_fn(self.input_layer(x))
        for i in range(self.depth):
            x = self.dropout(self.activation_fn(self.linears[i](x)))
        output = self.output_layer(x)
        return output
