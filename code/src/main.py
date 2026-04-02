import matplotlib.pyplot as plt
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

# runfile("/home/aloret/PycharmProjects/Neural_ST_inference/code/src/main.py")

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
nb_epochs = 400
freq_epochs_test = 10

# Data options

time_size = 30
space_dim = 1
space_size = 30
max_deepset_size = 5
missing_ratios = [0.92]

# Amount of data

train_size = 5
test_size = 5

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
# model = AdvectionModel()

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

# Creation of the several datasets used to test

dataset_test = GenericSimulationDataset(
    model=model,
    theta_size=test_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    max_deepset_size=max_deepset_size,
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

show_example(
    dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
)
# show_example_gif(
#     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}",num_time_steps=time_size, k=0
# )

# dataset_gridsearch = GenericSimulationDataset(
#     model=model,
#     theta_size=50,
#     time_size=time_size,
#     space_dim=space_dim,
#     space_size=space_size,
#     max_deepset_size=max_deepset_size,
#     missing_ratio=None,
# )
# print(f"\nData gridsearch generation : {model.__class__.__name__}")
# dataset_gridsearch.generate(data_path / "gridsearch")

############################
####### Architecture #######
############################

# # batch = dataset_train[0]
# # theta = batch[1][0]
# # batch = batch[0][0]
# # plt.imshow(batch[1])
# # plt.show()
# # # dataset_train.mle(batch)
# # print(dataset_train.kriging(batch.detach().numpy(), theta).shape)
#
# # Creation of the Neural Network
#
# CNN = CNN_ST_network(
#     in_channels=2,
#     out_dim=dataset_train.model.nb_params,
#     spatial_dim=space_dim,
# ).to(device)
#
# # Creation the Trainer used train, test and show results
#
# trainer_separately = Trainer(CNN, device, res_path)
#
# # Training (and testing at the same time)
#
# trainer_separately.train(
#     train_loader=loader_train,
#     test_loader=loaders_test[0],
#     nb_epochs=nb_epochs,
#     freq_epochs_test=freq_epochs_test,
#     print_result=True,
# )
#
# # Testing
#
# for loader in loaders_test:
#     trainer_separately.test(loader)

##################################
##################################
############ NLS Case ############
##################################
##################################

#####################
####### Model #######
#####################

# dataset_train_nls = NlsSimulationDataset(dataset_train, simu_nls=True)
# loader_train_nls = DataLoader(
#     dataset_train_nls, batch_size=batch_size, shuffle=True, collate_fn=deepset_collate
# )

# datasets_test_nls = []
# loaders_test_nls = []
# for ds in datasets_test :
#     dataset = NlsSimulationDataset(ds, simu_nls=False)
#     loaders_test_nls.append(DataLoader(
#         dataset, batch_size=1, shuffle=False, collate_fn=deepset_collate
#     ))
#
#     datasets_test_nls.append(dataset)

############################
####### Architecture #######
############################

# CNN = CNN_ST_network(
#     in_channels=2,
#     out_dim=dataset_train.model.nb_params,
#     spatial_dim=space_dim,
#     nls=True,
#     nls_sim=True,
# ).to(device)
# trainer_nls = Trainer(CNN, device, res_path)
# trainer_nls.train(
#     train_loader=loader_train_nls,
#     nb_epochs=nb_epochs,
#     print_result=True
# )
# for loader in loaders_test:
#     trainer_nls.test(loader, dataset_gridsearch)

###########################################
###########################################
############ Conclusion Graphs ############
###########################################
###########################################

# trainers = [trainer_separately]
#
# # train_graph(trainers, res_path)
# # time_hist(trainers, res_path)
# # test_alt_graph(trainers, res_path)
#
# for i in range(len(datasets_test)):
#     # test_graph(trainers, res_path, num_dataset=i)
#     # target_graph(trainers, res_path, k=0, num_dataset=i)
#     visualise(trainers, res_path, num_dataset=i)
