from torch.utils.data import DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

from utils import *
from models.gneiting_adv import GneitingAdvModel
from models.gneiting_sep import GneitingSepModel
from data.datasets import GenericSimulationDataset, NlsSimulationDataset
from architecture.train_test import *
from mle.mle_test import *
from architecture.neural_networks import CNN_ST_network, print_parameters_per_layer
from architecture.output_graphs import *
import warnings
import torch

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", module="pyro")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# runfile("/home/aloret/PycharmProjects/Neural_ST_inference/code/src/main.py")

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
# data_path = (Path("./output/data/spatiotemporal_adv").resolve() / "2026-09-10_10-00-57")  # "2026-08-03_15-23-11"

# Path to results saving file

res_path = create_res_path("output/results/spatiotemporal")
# res_path = (Path("./output/results/spatiotemporal").resolve() / "2026-09-10_10-00-57")  # "2026-08-03_15-23-11"

print("res_path", res_path)

# Neural Network train options

batch_size = 1000
nb_epochs = 20
freq_epochs_test = 10

# Data options

time_size = 100
space_dim = 1
space_size = 100
missing_ratios = [.94,.96,.98] # [0.94, 0.96, 0.98] # [0.7, 0.85, 0.95]

# Amount of data

train_size = 10
test_size = 10

if __name__ == "__main__":

    ##############################
    ##############################
    ############ Data ############
    ##############################
    ##############################

    ####################
    ####### Load #######
    ####################

    # model = GneitingAdvModel(space_dim=space_dim, noise=.5)
    #
    # dataset_train = GenericSimulationDataset.load(data_path / "train", model)
    # dataset_train.global_seed = 1
    # loader_train = DataLoader(
    #     dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True, persistent_workers=True
    # )
    #
    # datasets_test = []
    # loaders_test = []
    #
    # for ratio in missing_ratios:
    #     specific_test_path = data_path / "test"
    #
    #     dataset = GenericSimulationDataset.load(specific_test_path, model)
    #     dataset.missing_ratio = ratio
    #     # dataset.global_seed = 1+train_size*10
    #     print(dataset.missing_ratio)
    #     dataset.eval()
    #
    #
    #     datasets_test.append(dataset)
    #     loaders_test.append(
    #         DataLoader(dataset, batch_size=test_size, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
    #     )
    #
    #     # sum = 0
    #     # for _ in range(100):
    #     #     for i in range(len(dataset)):
    #     #         sum += torch.mean(dataset[i][0][0,1])
    #     # print("missing ratio", sum, len(dataset), sum/len(dataset))
    #
    #     # show_example_mask(
    #     #         dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
    #     #     )

    ########################
    ####### Generate #######
    ########################

    # Model choice

    model = GneitingAdvModel(space_dim=space_dim, noise=.5)
    # model = GneitingSepModel(space_dim=space_dim)

    # Creation of the class dataset and its θ

    dataset_train = GenericSimulationDataset(
        model=model,
        theta_size=train_size,
        time_size=time_size,
        space_dim=space_dim,
        space_size=space_size,
        missing_ratio=None,
        seed=1,
    )

    # Generation of data on a complete grid

    print(f"\nData train generation : {model.__class__.__name__}")
    dataset_train.generate(data_path / "train")

    # Creation of the loader used during training (or test)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    # Creation of the several datasets used to test

    dataset_test = GenericSimulationDataset(
        model=model,
        theta_size=test_size,
        time_size=time_size,
        space_dim=space_dim,
        space_size=space_size,
        missing_ratio=None,
        seed=1+train_size*10,
    )
    dataset_test.generate(data_path / "test")
    dataset_test.eval()

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
                batch_size=test_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=4,
                pin_memory=True,
            )
        )

    # Show data example

    # show_example_mask(
    #     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
    # )
    # show_example_mask(
    #     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
    # )
    # show_example_mask(
    #     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}", k=5
    # )
    # show_example_mask(
    #     datasets_test[0], data_path / "test" / f"ratio_{missing_ratios[0]}", k=0
    # )
    # show_example_gif(
    #     dataset_train, data_path / "test" / f"ratio_{missing_ratios[0]}",num_time_steps=time_size, k=0
    # )

    #############################
    #############################
    ############ NBE ############
    #############################
    #############################

    ############################
    ####### Architecture #######
    ############################

    mle_trainer = MLE(device, res_path)
    mle_trainer.test_mle(datasets_test[0])

    # Creation of the Neural Network

    CNN = CNN_ST_network(
        in_channels=2,
        hidden=72,
        theta_dim=dataset_train.model.nb_params,
        spatial_dim=space_dim,
        label="NBE",
    ).to(device)

    print_parameters_per_layer(CNN)

    quantile_CNN = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=dataset_train.model.nb_params,
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)

    # Creation the Trainer used train, test and show results

    trainer_separately = Trainer(
        CNN,
        device,
        res_path,
        quantiles=[0.05, 0.95],
        quantile_networks=[copy.deepcopy(quantile_CNN), copy.deepcopy(quantile_CNN)],
    )

    # Training (and testing at the same time)

    trainer_separately.train(
        train_loader=loader_train,
        test_loader=loaders_test[0],
        nb_epochs=nb_epochs,
        freq_epochs_test=freq_epochs_test,
    )

    # Testing

    for loader in loaders_test:
        res = trainer_separately.test(loader)  # res = ...

    try: # 0.8960, 0.9113, 0.8900, 0.9113
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

    # #############################
    # #############################
    # ############ NRE ############
    # #############################
    # #############################
    #
    # #####################
    # ####### Model #######
    # #####################
    #
    # dataset_train_nls = NlsSimulationDataset(dataset_train)
    # loader_train_nls = DataLoader(
    #     dataset_train_nls, batch_size=batch_size, shuffle=True, collate_fn=collate,
    #     num_workers=4,
    #     pin_memory=True
    # )
    #
    # ############################
    # ####### Architecture #######
    # ############################
    #
    # CNN = CNN_ST_network(
    #     in_channels=2,
    #     hidden=72,
    #     theta_dim=dataset_train.model.nb_params,
    #     spatial_dim=space_dim,
    #     nls=True,
    #     label="NRE"
    # ).to(device)
    #
    # print_parameters_per_layer(CNN)
    #
    # trainer_nls = Trainer(
    #     CNN,
    #     device,
    #     res_path,
    #     quantiles=[.05, .5, .95]
    # )
    # trainer_nls.train(
    #     train_loader=loader_train_nls,
    #     nb_epochs=nb_epochs,
    # )
    # for loader in loaders_test:
    #     res = trainer_nls.test(loader)
    #
    # try :
    #     print(
    #         "quantiles precision :",
    #         torch.mean(
    #             (res["true_all_theta"] > res["all_quantiles"][0.05])
    #             * (res["true_all_theta"] < res["all_quantiles"][0.95]).float(),
    #             dim=0,
    #         ),
    #     )
    # except:
    #     print("No quantiles")

    # ###########################################
    # ###########################################
    # ############ Conclusion Graphs ############
    # ###########################################
    # ###########################################

    # visualise([trainer_separately], res_path, num_dataset=0, nb_examples=2)

    trainers = [trainer_separately]

    # train_graph(trainers, res_path)
    time_hist(trainers, res_path)
    test_alt_graph(trainers, res_path)

    for i in range(len(datasets_test)):
        test_graph(trainers, res_path, num_dataset=i)
        # target_graph(trainers, res_path, k=0, num_dataset=i)
        # visualise(trainers, res_path, num_dataset=i, nb_examples=2)
