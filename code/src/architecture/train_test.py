import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmin

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import map_deepset, baseline_mae, THETA_COLORS


def train_test(architecture, optimizer, train_loader, test_loader, nb_epochs, freq_epochs_test, device,
               res_path, print_result=False):

    all_loss_train = []
    all_loss_test = []

    loss_final_train = None
    loss_final_test = None

    t = tqdm(range(nb_epochs), total=nb_epochs)

    time_train_start = time.time()
    time_test_accumulated = 0

    MAE = torch.nn.L1Loss(reduction="none")

    for epoch in t:
        architecture.train()

        epoch_batch_means = []

        epoch_full_errors = []

        for i, (batched_data, theta, deepset) in enumerate(train_loader):
            optimizer.zero_grad()

            batched_data = batched_data.to(device).float()
            theta = theta.to(device)
            deepset = deepset.to(device)

            deepset_mapped, theta = map_deepset(deepset, theta, device)

            output = architecture(batched_data, deepset_mapped)

            loss_vec = MAE(output, theta)

            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            epoch_batch_means.append(loss_vec.mean(dim=0).detach().cpu())

            if epoch == nb_epochs - 1:
                epoch_full_errors.append(loss_vec.detach().cpu())

        mean_epoch = torch.stack(epoch_batch_means).mean(dim=0)
        all_loss_train.append(mean_epoch)

        if epoch == nb_epochs - 1:
            loss_final_train = torch.cat(epoch_full_errors, dim=0)

        loss_str = ", ".join([f"{l:.4f}" for l in mean_epoch])
        t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
        t.set_postfix_str(f"Loss: [{loss_str}]")

        if (epoch + 1) % freq_epochs_test == 0:
            start_test = time.time()
            architecture.eval()

            test_errors_list = []

            with torch.no_grad():
                for data_test, theta_test, deepset_test in test_loader:
                    data_test = data_test.to(device).float()
                    theta_test = theta_test.to(device)
                    deepset_test = deepset_test.to(device)

                    deepset_mapped_test, theta_test = map_deepset(deepset_test, theta_test, device)

                    output = architecture(data_test, deepset_mapped_test)

                    errors = torch.abs(output - theta_test)
                    test_errors_list.append(errors.detach().cpu())

            full_test_errors = torch.cat(test_errors_list, dim=0)

            all_loss_test.append(full_test_errors.mean(dim=0))

            loss_final_test = full_test_errors

            time_test_accumulated += (time.time() - start_test)

    total_time_train = (time.time() - time_train_start) - time_test_accumulated

    if print_result:
        print(f"\nTime to train : {total_time_train:.2f}s")

        last_train = all_loss_train[-1].tolist()
        print("Train loss per param : " + train_loader.dataset.model.to_title(last_train, alt_format=True))

        if all_loss_test:
            last_test = all_loss_test[-1].tolist()
            print("Test loss per param : " + test_loader.dataset.model.to_title(last_test, alt_format=True))

    results = {
        "train_loss_history": torch.stack(all_loss_train),  # (Nb_epochs, Nb_params)
        "test_loss_history": torch.stack(all_loss_test) if all_loss_test else None,  # (Nb_tests, Nb_params)
        "train_time": total_time_train,
        "test_time": time_test_accumulated,
        "final_train_losses": loss_final_train,  # (Size_Train, Nb_params)
        "final_test_losses": loss_final_test  # (Size_Test, Nb_params)
    }

    save_path = res_path / "loss_evolution" / f"{architecture.label}_history.pt"
    res_path.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)

    _plot_loss(all_loss_train, all_loss_test, test_loader, res_path, nb_epochs, freq_epochs_test, architecture.label)

    torch.save(architecture.state_dict(), res_path / "architectures" / f"{architecture.label}_architecture.pth")


def test(architecture, loader_test, device, print_result=False):
    all_loss_test = []
    all_theta_final = []

    time_start = time.time()
    architecture.eval()

    model_ref = loader_test.dataset.model
    t = tqdm(loader_test, total=len(loader_test), desc=f"Inference {architecture.label}")

    with torch.no_grad():
        for i, (data_test, theta_test, deepset_test) in enumerate(t):
            data_test = data_test.to(device).float()
            theta_test = theta_test.to(device)
            deepset_test = deepset_test.to(device)

            deepset_mapped, theta_target = map_deepset(deepset_test, theta_test, device)
            output = architecture(data_test, deepset_mapped)

            loss = torch.abs(output - theta_target).detach().cpu()
            theta_final = output.detach().cpu()

            all_loss_test.append(loss)
            all_theta_final.append(theta_final)

            if i % 5 == 0:
                current_mae = torch.cat(all_loss_test).mean().item()
                t.set_postfix(Loss=f"{current_mae:.4f}")

    time_total = time.time() - time_start
    full_loss_tensor = torch.cat(all_loss_test, dim=0)
    print(all_theta_final)
    all_theta_final = torch.cat(all_theta_final, dim=0)

    if print_result:
        mean_loss = full_loss_tensor.mean(dim=0)
        print(f"Time to test : {time_total:.2f}s")
        print(f"Test Loss (MAE) per param: " + model_ref.to_title(mean_loss, alt_format=True))

    return full_loss_tensor, all_theta_final, time_total


def _plot_loss(all_loss_train, all_loss_test, test_loader, res_path, nb_epochs, freq_epochs_test, label):
    theta_names = test_loader.dataset.model.param_names

    fig, ax = plt.subplots(1, len(theta_names), figsize=(20, 7), constrained_layout=True)
    based, _ = baseline_mae(test_loader.dataset)

    for i, name in enumerate(theta_names):
        ax[i].plot([l[i] for l in all_loss_train], THETA_COLORS[i]['train'], label='train ' + name)
        ax[i].plot(range(freq_epochs_test, nb_epochs + 1, freq_epochs_test), [l[i] for l in all_loss_test],
                   THETA_COLORS[i]['test'], label='test ' + name)

        ax[i].axhline(based[i], color=THETA_COLORS[i]['baseline'], ls='--', label='baseline ' + name)

        ax[i].set_title('Evolution of ' + name)
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel('MAE')
        ax[i].legend()

    fig.suptitle('Loss evolutions of ' + label)

    fig.savefig(res_path / "loss_evolution" / f"{label}_train_evolution.png", format="png", bbox_inches="tight")
    fig.savefig(res_path / "loss_evolution" / f"{label}_train_evolution.pdf", format="pdf", bbox_inches="tight")


def train_nls(architecture, optimizer, train_loader, nb_epochs, device, res_path, print_result=False):

    all_loss_train = []

    t = tqdm(range(nb_epochs), total=nb_epochs)

    time_train_start = time.time()

    CE = nn.CrossEntropyLoss()

    for epoch in t:
        architecture.train()

        epoch_batch = []

        for i, (batched_data, theta, classi) in enumerate(train_loader):
            optimizer.zero_grad()

            batched_data = batched_data.to(device).float()
            theta = theta.to(device)
            classi = classi.to(device)

            output = architecture(batched_data, theta=theta)

            loss = CE(output, classi).mean()

            loss.backward()
            optimizer.step()

            epoch_batch.append(loss.detach().cpu())

        mean_epoch = torch.stack(epoch_batch).mean()
        all_loss_train.append(mean_epoch)

        t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
        t.set_postfix_str(f"Loss: {mean_epoch.item():.3f}")

    total_time_train = time.time() - time_train_start

    if print_result:
        print(f"\nTime to train : {total_time_train:.2f}s")

        last_train = all_loss_train[-1].tolist()
        print("Train loss : " + str(last_train))

    results = {
        "train_loss_history": torch.stack(all_loss_train),  # (Nb_epochs, Nb_params)
        "train_time": total_time_train,
    }

    save_path = res_path / "loss_evolution" / f"{architecture.label}_history.pt"
    res_path.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
    ax.plot(all_loss_train, 'forestgreen', linewidth=2.)

    fig.suptitle('Loss evolutions of ' + architecture.label)
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cross Entropy')

    fig.savefig(res_path / "loss_evolution" / f"{architecture.label}_train_evolution.png", format="png",
                bbox_inches="tight")
    fig.savefig(res_path / "loss_evolution" / f"{architecture.label}_train_evolution.pdf", format="pdf",
                bbox_inches="tight")

    torch.save(architecture.state_dict(), res_path / "architectures" / f"{architecture.label}_architecture.pth")


def _nls_optim(data, architecture_nls, theta0, device, activation=lambda x:x):

    def obj(theta):
        theta_t = activation(theta.unsqueeze(0).to(device))
        # theta_t = theta.unsqueeze(0).to(device)
        h = architecture_nls(data, theta=theta_t)

        logL = h[:,1] - h[:,0]
        return logL

    res = torchmin.minimize(
        obj,
        theta0,
        method="bfgs",
        options={
            "max_iter": 500,
            "lr": 0.2,
        },
        return_all=True
    )
    return activation(res.x), activation(res.allvecs[-2])

def test_nls(architecture_nls, loader_test, device, theta0=None, print_result=False):

    assert loader_test.batch_size == 1, "loader_test has to be 1 in NLS case"

    all_loss_test = []
    all_theta_final = []

    time_start = time.time()

    model_ref = loader_test.dataset.model
    lower_bound = torch.tensor(model_ref.lower_bound).to(device)
    upper_bound = torch.tensor(model_ref.upper_bound).to(device)

    if theta0 is None:
        theta0 = (upper_bound - lower_bound)/2
    activation = lambda x:torch.sigmoid(x) * upper_bound-lower_bound + lower_bound

    t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

    for i, (data_test, theta_test, classi) in enumerate(t):

        if classi :
            continue

        data_test = data_test.to(device).float()
        true_theta = theta_test.to(device)

        output, output_not_nan = _nls_optim(data_test, architecture_nls, theta0, device, activation)

        loss = torch.abs(output - true_theta).detach().cpu()
        theta_final = output.detach().cpu()

        if torch.isnan(loss).any():
            loss = torch.abs(output_not_nan.cpu() - true_theta.cpu())
            theta_final = output_not_nan.detach().cpu()

        all_loss_test.append(loss)
        all_theta_final.append(theta_final.unsqueeze(0))

        if i % 5 == 0:
            current_mae = torch.cat(all_loss_test).mean().item()
            t.set_postfix(Loss=f"{current_mae:.4f}")

    time_total = time.time() - time_start
    full_loss_tensor = torch.cat(all_loss_test, dim=0)
    print(all_theta_final)
    all_theta_final = torch.cat(all_theta_final, dim=0)

    mean_loss = full_loss_tensor.mean(dim=0)
    std_loss = full_loss_tensor.std(dim=0)

    if print_result:
        print(f"Time to test : {time_total:.2f}s")

        print(f"Test Loss (MAE) per param: "+model_ref.to_title(mean_loss, alt_format=True))

    return full_loss_tensor, all_theta_final, time_total