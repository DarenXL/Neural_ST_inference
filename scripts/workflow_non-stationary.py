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

data_path = create_data_path("other_codes/data/spatiotemporal_adv")
# data_path = Path("./other_codes/data/spatiotemporal_adv").resolve() / "2D-try"

# Path to results saving file

res_path = create_res_path("other_codes/results/spatiotemporal")

# Neural Network train options

batch_size = 40
nb_epochs = 400
freq_epochs_test = 10

# Data options

time_size = 30
space_dim = 1
space_size = 30
max_deepset_size = 5
missing_ratios = [0.70, .85, .92]

# Amount of data

train_size = 1
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

model = GneitingSepModel(space_dim=space_dim)

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

theta_droite = np.array([0.0000, 0.15, 0.06])
theta_gauche = np.array([0.0000, 0.10, 0.15])
img = dataset_train.generate_from_theta(theta_droite)[0,0]
mask = np.concatenate([np.ones((15,30)), np.zeros((15,30))], axis=0)
img = np.stack((img, mask))
krig = dataset_train.kriging_from_sample(img, theta_gauche)
simu = img.copy()
simu[0] = dataset_train.generate_from_theta(theta_gauche)
krig_simu = dataset_train.kriging_from_sample(simu, theta_gauche)
plt.imshow(simu[0] - krig_simu + krig, cmap=CMAP)
plt.ylabel(r"Time")
plt.xlabel(r"Space")
plt.title("Non-stationary example")
plt.show()
# plt.savefig(data_path / "train" / "examples" / f"non-stationary.pdf", format="pdf", bbox_inches="tight")
