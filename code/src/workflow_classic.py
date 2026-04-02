from torch.utils.data import Dataset, DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

from utils import *
from models.advection import AdvectionModel
from models.gneiting_adv import GneitingAdvModel
from models.gneiting_sep import GneitingSepModel
from data.datasets import GenericSimulationDataset, NlsSimulationDataset
from architecture.train_test import *
from architecture.neural_networks import CNN_ST_network
from architecture.output_graphs import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#########################
####### Variables #######
#########################

# Path to data saving file

data_path = create_data_path("code/data/spatiotemporal_adv")
# data_path = Path("./code/data/spatiotemporal_adv").resolve() / "2D-try"

# Path to results saving file

res_path = create_res_path("code/results/spatiotemporal")

# Neural Network train options

batch_size = 40
nb_epochs = 40
freq_epochs_test = 10

# Data options

time_size = 30
space_dim = 1
space_size = 30
max_deepset_size = 5
missing_ratios = [0.70, .85, .92]

# Amount of data

train_size = 20
test_size = 4

######################################
######################################
############ Generic Case ############
######################################
######################################

#####################
####### Model #######
#####################

# Model choice

model = GneitingAdvModel(space_dim=space_dim)

# Creation of the class dataset and its θ

dataset_train = GenericSimulationDataset(
    model=model,
    theta_size=train_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    max_deepset_size=max_deepset_size,
    missing_ratio=None,
)

# Generation of data on a complete grid

print(f"\nData train generation : {model.__class__.__name__}")
dataset_train.generate(data_path / "train")

# Creation of the loader used during training (or test)

loader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=deepset_collate
)

# Creation of the dataset used to test

dataset_test = GenericSimulationDataset(
    model=model,
    theta_size=test_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    max_deepset_size=5,
    missing_ratio=None,
)
dataset_test.generate(data_path / "test")

datasets_test = []
loaders_test = []

for ratio in missing_ratios:

    dataset = copy.deepcopy(dataset_test)
    dataset.missing_ratio = ratio

    datasets_test.append(dataset)
    loaders_test.append(
        DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=deepset_collate)
    )

# Show data example

# show_example(
#     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
# )
# show_example_gif(
#     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}",num_time_steps=time_size, k=0
# )

############################
####### Architecture #######
############################

# Creation of the Neural Network

CNN = CNN_ST_network(
    in_channels=2,
    out_dim=dataset_train.model.nb_params,
    spatial_dim=space_dim,
    pool_option="max"
).to(device)

# Creation the Trainer used train, test and show results

trainer = Trainer(CNN, device, res_path)

# Training (and testing at the same time)

trainer.train(
    train_loader=loader_train,
    test_loader=loaders_test[0],
    nb_epochs=nb_epochs,
    freq_epochs_test=freq_epochs_test,
    print_result=True,
)

# Testing

for loader in loaders_test:
    trainer.test(loader)

###########################################
###########################################
############ Conclusion Graphs ############
###########################################
###########################################

trainers = [trainer]

train_graph(trainers, res_path)
time_hist(trainers, res_path)
test_alt_graph(trainers, res_path)

for i in range(len(datasets_test)):
    test_graph(trainers, res_path, num_dataset=i)
    target_graph(trainers, res_path, k=0, num_dataset=i)
    visualise(trainers, res_path, num_dataset=i)

