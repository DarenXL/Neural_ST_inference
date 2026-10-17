from torch.utils.data import DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

from neural_st_inference.utils import *
from neural_st_inference.models.gneiting_adv import GneitingAdvModel
from neural_st_inference.data.datasets import GenericSimulationDataset, ClassifierSimulationDataset
from neural_st_inference.architecture.train_test import *
from neural_st_inference.architecture.neural_networks import CNN_ST_network, print_parameters_per_layer
from neural_st_inference.output_graphs import *
from neural_st_inference.mle.mle_test import *

import warnings
import torch

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", module="pyro")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# runfile("/home/aloret/PycharmProjects/Neural_ST_inference/other_codes/src/main.py")

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        # "font.family": "inconsolata"
    }
)

np.random.seed(2)
torch.manual_seed(2)


#########################
####### Variables #######
#########################

# Path to data saving file

data_path = create_data_path("output/data/spatiotemporal_adv")
# data_path = (Path("./output/data/spatiotemporal_adv").resolve() / "2026-09-10_10-00-57")

# Path to results saving file

res_path = create_res_path("output/results/spatiotemporal")
# res_path = (Path("./output/results/spatiotemporal").resolve() / "2026-10-08_09-55-51")

print("res_path", res_path)

# Neural Network train options

batch_size_train = 512
batch_size_test = 2000
nb_epochs = 20
freq_epochs_test = 10

# Data options

time_size = 20
space_dim = 1
space_size = 20
missing_ratios = [0.96] # [0.7, 0.85, 0.92] # [0.94, 0.96, 0.98]   # ATTENTION DANS UTILS.PY
dev, noise = 1.414, .5 # 1.414, .5

# Amount of data

train_size = 5
test_size = 5

if __name__ == "__main__":

    ##############################
    ##############################
    ############ Data ############
    ##############################
    ##############################

    ####################
    ####### Load #######
    ####################

    # # Model
    #
    # model = GneitingAdvModel(space_dim=space_dim)
    #
    # # Training dataset
    #
    # dataset_train = GenericSimulationDataset.load(data_path / "train", model)
    #
    # loader_train = DataLoader(
    #     dataset_train, batch_size=batch_size_train, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True
    # )
    #
    # # Test datasets
    #
    # datasets_test = []
    # loaders_test = []
    #
    # for ratio in missing_ratios:
    #     specific_test_path = data_path / "test"
    #
    #     dataset = GenericSimulationDataset.load(specific_test_path, model)
    #     dataset.missing_ratio = ratio
    #
    #     dataset.eval()
    #
    #     datasets_test.append(dataset)
    #     loaders_test.append(
    #         DataLoader(dataset, batch_size=batch_size_test, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
    #     )

    ########################
    ####### Generate #######
    ########################

    # Model

    model = GneitingAdvModel(space_dim=space_dim)

    # Training dataset

    dataset_train = GenericSimulationDataset(
        model=model,
        theta_size=train_size,
        time_size=time_size,
        space_dim=space_dim,
        space_size=space_size,
        missing_ratio=None,
        seed=1,
        dev=dev,
        noise=noise
    )

    dataset_train.generate(data_path / "train")

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    # Test dataset base

    dataset_test = GenericSimulationDataset(
        model=model,
        theta_size=test_size,
        time_size=time_size,
        space_dim=space_dim,
        space_size=space_size,
        missing_ratio=None,
        seed=1+train_size*10,
        dev=dev,
        noise=noise
    )
    dataset_test.generate(data_path / "test")
    dataset_test.eval()

    # Test dataset for each missing ratio

    datasets_test = []
    loaders_test = []

    for ratio in missing_ratios:

        dataset = copy.deepcopy(dataset_test)
        dataset.missing_ratio = ratio
        dataset.generate_masks(data_path / "test")

        datasets_test.append(dataset)
        loaders_test.append(
            DataLoader(
                dataset,
                batch_size=batch_size_test,
                shuffle=False,
                collate_fn=collate,
                num_workers=4,
                pin_memory=True,
            )
        )

    # Show data example

    # show_example_mask(
    #     datasets_test[0], data_path / "test" / f"ratio_{missing_ratios[0]}", k=4
    # )

    #############################
    #############################
    ############ NBE ############
    #############################
    #############################

    # Creation of the neural network and the quantile networks

    CNN = CNN_ST_network(
        in_channels=2,
        hidden=72,
        theta_dim=len(dataset_train.theta_names),
        spatial_dim=space_dim,
        label="NBE",
    ).to(device)
    # print_parameters_per_layer(CNN)

    quantile_CNN = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=len(dataset_train.theta_names),
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)

    # Creation the Trainer used train, test and show results

    trainer_nbe = Trainer(
        CNN,
        device,
        res_path,
        quantiles=[0.05, 0.95],
        quantile_networks=[copy.deepcopy(quantile_CNN), copy.deepcopy(quantile_CNN)],
    )

    # Training (and testing at the same time)

    trainer_nbe.train(
        train_loader=loader_train,
        test_loader=loaders_test[0],
        nb_epochs=nb_epochs,
        freq_epochs_test=freq_epochs_test,
    )

    # Testing

    for loader in loaders_test:
        res = trainer_nbe.test(loader)  # res = ...

    try:
        print(
            "quantiles precision :",
            torch.mean(
                (res["true_all_theta"] > res["all_quantiles"][0.05])
                * (res["true_all_theta"] < res["all_quantiles"][0.95]).float(),
                dim=0,
            ),
        )
    except:
        print("No quantiles")

    # Prediction score

    score = trainer_nbe.pred_error(res["final_all_theta"])["prediction_scores"]
    print("Prediction score :", score.mean(axis=0))

    #############################
    #############################
    ############ NRE ############
    #############################
    #############################

    # Dataset adapted to classifiers

    dataset_train_nls = ClassifierSimulationDataset(dataset_train)
    loader_train_nls = DataLoader(
        dataset_train_nls, batch_size=batch_size_train, shuffle=True, collate_fn=collate,
        num_workers=4,
        pin_memory=True
    )

    # Creation of the neural network

    CNN = CNN_ST_network(
        in_channels=2,
        hidden=72,
        theta_dim=len(dataset_train.theta_names),
        spatial_dim=space_dim,
        nls=True,       # NRE parameter
        label="NRE"
    ).to(device)
    # print_parameters_per_layer(CNN)

    # Creation the Trainer used train, test and show results

    trainer_nre = Trainer(
        CNN,
        device,
        res_path,
        quantiles=[.05, .5, .95]
    )

    # Training

    trainer_nre.train(
        train_loader=loader_train_nls,
        nb_epochs=nb_epochs,
    )

    # Testing

    for loader in loaders_test:
        res = trainer_nre.test(loader)

    try :
        print(
            "quantiles precision :",
            torch.mean(
                (res["true_all_theta"] > res["all_quantiles"][0.05])
                * (res["true_all_theta"] < res["all_quantiles"][0.95]).float(),
                dim=0,
            ),
        )
    except:
        print("No quantiles")

    score = trainer_nre.pred_error(res["final_all_theta"])["prediction_scores"]
    print("Prediction score :", score.mean(axis=0))

    #############################
    #############################
    ############ MLE ############
    #############################
    #############################

    mle_trainer = MLE(device, res_path)
    res = mle_trainer.test_mle(datasets_test[0])

    score = mle_trainer.pred_error(res["final_all_theta"])["prediction_scores"]
    print("Prediction score :", score.mean(axis=0))

    ###########################################
    ###########################################
    ############ Conclusion Graphs ############
    ###########################################
    ###########################################

    trainers = [trainer_nbe, trainer_nre, mle_trainer]

    # Method's time comparison

    time_hist(trainers, res_path)

    # Method's score comparison

    for i in range(len(datasets_test)):
        test_graph(trainers, res_path, num_dataset=i)

    # Missing ratio comparison

    test_alt_graph(trainers, res_path)

    # Kriging Visualisation

    visualise([trainer_nbe], res_path, num_dataset=0, nb_examples=2, k=[4, 7])






