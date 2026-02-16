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
from data.datasets import GenericSimulationDataset, NlsSimulationDataset
from architecture.train_test import *
from architecture.neural_networks import CNN_ST_network
from architecture.output_graphs import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#TODO : commenter le code

#############
# Variables #
#############

# Paths
data_path = create_data_path("./code/data/spatiotemporal_adv")
# data_path = Path("./code/data/spatiotemporal_adv").resolve() / "2D-try"

res_path = create_res_path("./code/results/spatiotemporal")

# Neural Network train options
batch_size = 50
nb_epochs = 50
freq_epochs_test = 10

time_size = 10
space_dim = 1
space_size = 10

deepset_size = 5
missing_ratio = 1 / 2

train_size = 50
test_size = 10

####################
####################
### Generic Case ###
####################
####################

#########
# Model #
#########

model = AdvectionModel()
# model = GneitingModel()

dataset_train = GenericSimulationDataset(
    model=model,
    theta_size=train_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    deepset_size=deepset_size,
    missing_ratio=missing_ratio,
)

print(f"Data train generation : {model.__class__.__name__}")
dataset_train.generate(data_path / "train")

dataset_test = GenericSimulationDataset(
    model=model,
    theta_size=test_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    deepset_size=5,
    missing_ratio=missing_ratio,
)

print(f"Data test generation : {model.__class__.__name__}")
dataset_test.generate(data_path / "test")

# dataset_train = GenericSimulationDataset.load(data_path / "train", model_instance=model)
# dataset_test = GenericSimulationDataset.load(data_path / "test", model_instance=model)

loader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=deepset_collate
)
loader_test = DataLoader(
    dataset_test, batch_size=5, shuffle=False, collate_fn=deepset_collate
)

# show_example(loader_test, k=1)

baseline, theta_mean = baseline_mae(dataset_test)

################
# Architecture #
################

CNN = CNN_ST_network(
    in_channels=2,
    out_dim=dataset_train.model.nb_params,
    spatial_dim=space_dim,
    option="jointly",
).to(device)
trainer_jointly = Trainer(CNN, device, res_path)
trainer_jointly.train(
    train_loader=loader_train,
    test_loader=loader_test,
    nb_epochs=nb_epochs,
    freq_epochs_test=freq_epochs_test,
    print_result=True,
)
trainer_jointly.test(loader_test)

CNN = CNN_ST_network(
    in_channels=2,
    out_dim=dataset_train.model.nb_params,
    spatial_dim=space_dim,
    option="separately",
).to(device)
trainer_separately = Trainer(CNN, device, res_path)
trainer_separately.train(
    train_loader=loader_train,
    test_loader=loader_test,
    nb_epochs=nb_epochs,
    freq_epochs_test=freq_epochs_test,
    print_result=True,
)
trainer_separately.test(loader_test)

################
################
### NLS Case ###
################
################

#########
# Model #
#########

# dataset_train_nls = NlsSimulationDataset(dataset_train)
# dataset_test_nls = NlsSimulationDataset(dataset_test)
#
# loader_train_nls = DataLoader(
#     dataset_train_nls, batch_size=batch_size, shuffle=True, collate_fn=deepset_collate
# )
# loader_test_nls = DataLoader(
#     dataset_test_nls, batch_size=5, shuffle=False, collate_fn=deepset_collate
# )

################
# Architecture #
################

# CNN = CNN_ST_network(
#     in_channels=2,
#     out_dim=dataset_train.model.nb_params,
#     spatial_dim=space_dim,
#     option="jointly",
#     nls=True,
# ).to(device)
# trainer_nls = Trainer(CNN, device, res_path)
# trainer_nls.train(
#     train_loader=loader_train_nls,
#     nb_epochs=nb_epochs,
#     print_result=True
# )
# trainer_nls.test(loader_test_nls)

#########################
#########################
### Conclusion Graphs ###
#########################
#########################

trainers = [trainer_jointly, trainer_separately] #, trainer_nls]

train_graph(trainers, res_path)
test_graph(trainers, res_path)
time_hist(trainers, res_path)
target_graph(trainers, res_path, k=0)

visualise_output(trainer_jointly, res_path)

