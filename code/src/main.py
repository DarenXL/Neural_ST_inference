import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import numpy as np

import random
from tqdm import tqdm
import time
import os
import torchmin
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import scipy.linalg as linalg
from scipy.special import gamma, kv

import gstlearn as gl
import gstlearn.plot as gp
import gstlearn.document as gdoc
gdoc.setNoScroll()

from utils import *
from models.advection import AdvectionModel
from models.gneiting import GneitingModel
from data.datasets import GenericSimulationDataset
from architecture.train_test import train_test
from architecture.neural_networks import CNN_ST_network

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#############
# Variables #
#############

# Paths
# data_path = create_data_path("./code/data/spatiotemporal_adv")
data_path = Path("./code/data/spatiotemporal_adv").resolve() / "3D-try"

res_path = create_res_path("./code/results/spatiotemporal")

# Neural Network train options
batch_size = 50
nb_epochs = 50
freq_epochs_test = 10

time_size = 10
space_dim = 2
space_size = 20

deepset_size = 2
missing_ratio = 1/2

train_size = 50
test_size = 10

#########
# Model #
#########

model = AdvectionModel()
# model = GneitingModel()

# dataset_train = GenericSimulationDataset(
#     model=model,
#     theta_size=train_size,
#     time_size=time_size,
#     space_dim=space_dim,
#     space_size=space_size,
#     deepset_size=deepset_size,
#     missing_ratio=missing_ratio,
# )
#
# print(f"Data train generation : {model.__class__.__name__}")
# dataset_train.generate(data_path / "train")
#
# dataset_test = GenericSimulationDataset(
#     model=model,
#     theta_size=test_size,
#     time_size=time_size,
#     space_dim=space_dim,
#     space_size=space_size,
#     deepset_size=1,
#     missing_ratio=missing_ratio,
# )
#
# print(f"Data test generation : {model.__class__.__name__}")
# dataset_test.generate(data_path / "test")

dataset_train = GenericSimulationDataset.load(data_path / "train", model_instance=model)
dataset_test = GenericSimulationDataset.load(data_path / "test", model_instance=model)

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,collate_fn=deepset_collate)
loader_test = DataLoader(dataset_test, batch_size=10, shuffle=False,collate_fn=deepset_collate)

# show_example(dataset_test, loader_test, k=1)

################
# Architecture #
################

CNN = CNN_ST_network(in_channels=2, out_dim=dataset_train.model.nb_params, spatial_dim=space_dim, option="jointly").to(device)
optimizer = torch.optim.Adam(CNN.parameters(), lr=5e-5)
train_test(CNN, optimizer, loader_train, loader_test, nb_epochs, freq_epochs_test, device, res_path, True)

CNN= CNN_ST_network(in_channels=2, out_dim=dataset_train.model.nb_params, spatial_dim=space_dim, option="separately").to(device)
optimizer= torch.optim.Adam(CNN.parameters(), lr=5e-5)
train_test(CNN, optimizer, loader_train, loader_test, nb_epochs, freq_epochs_test, device, res_path, True)
