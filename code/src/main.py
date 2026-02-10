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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#############
# Variables #
#############

# Paths
data_path = create_data_path("./code/data/spatiotemporal_2D")
res_path = create_res_path("./code/results/spatiotemporal_2D")

# Neural Network train options
batch_size = 50
n_epochs = 250
n_epochs_test = 10

time_size = 30
space_dim = 1
space_size = 30

deepset_size = 2
missing_ratio = 1/2

train_size = 16
test_size = 5

#########
# Model #
#########

model = AdvectionModel()
# model = GneitingModel()

dataset_train = GenericSimulationDataset(
    model=model,
    theta_size=train_size,
    time_size=time_size,
    space_size=space_size,
    deepset_size=deepset_size,
    missing_ratio=missing_ratio,
)

print(f"Data generation : {model.__class__.__name__}")
dataset_train.generate(data_path)

data, theta, deepset_idx = dataset_train[1]
# print("Shape data:", data.shape)
# print("Theta:", theta)
# print("Deepset:", deepset_idx)

plt.imshow(data[0,0])
plt.show()
