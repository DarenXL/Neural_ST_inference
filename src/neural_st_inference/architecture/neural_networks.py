import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_st_inference.utils import *


class CNN_ST_network(nn.Module):
    def __init__(
        self,
        in_channels=2,
        hidden=32,
        theta_dim=4,
        spatial_dim=1,
        pool_option="max",
        nls=False,
        label=None,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.nls = nls

        if label is None:
            self.label = f"{pool_option} {spatial_dim}D-Space CNN" + (
                " nls" if nls else ""
            )
        else:
            self.label = label

        if spatial_dim == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden // 2, kernel_size=(5, 8), padding=(2, 3)),
                nn.ReLU(),
                nn.Conv2d(hidden // 2, hidden // 2, kernel_size=(5, 5), padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(hidden // 2, hidden, kernel_size=(3, 5), padding=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            # self.conv = nn.Sequential(
            #     nn.Conv2d(in_channels, hidden, kernel_size=(8, 8), padding=2),
            #     nn.ReLU(),
            #     nn.Conv2d(hidden, hidden, kernel_size=(5, 5), padding=1),
            #     nn.ReLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(hidden, hidden, kernel_size=(5, 5), padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            #     nn.ReLU(),
            # )

            if pool_option == "avg":
                self.pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.pool = nn.AdaptiveMaxPool2d(1)

        elif spatial_dim == 2:
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, hidden // 2, kernel_size=(5, 8, 8), padding=(2, 3, 3)
                ),
                nn.ReLU(),
                nn.Conv3d(
                    hidden // 2, hidden // 2, kernel_size=(5, 5, 5), padding=(2, 2, 2)
                ),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    hidden // 2, hidden, kernel_size=(3, 5, 5), padding=(1, 2, 2)
                ),
                nn.ReLU(),
                nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ReLU(),
            )
            if pool_option == "avg":
                self.pool = nn.AdaptiveAvgPool3d(1)
            else:
                self.pool = nn.AdaptiveMaxPool3d(1)

        self.dense = nn.Sequential(
            nn.Linear(hidden + (theta_dim * self.nls), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1 if self.nls else theta_dim),
        )

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta=None):

        B = x.shape[0]
        T = x.shape[2]

        summary = self.conv(x)
        summary = self.pool(summary)
        summary = summary.view(B, -1)

        if self.nls and theta is not None:
            summary = torch.cat((summary, theta.float()), dim=1)

        out = self.dense(summary)

        return out


def print_parameters_per_layer(model):
    print(f"{'Layer Name':<25} | {'Parameters':<10}")
    print("-" * 40)
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.parameters(recurse=False))) > 0:
            params = sum(p.numel() for p in module.parameters(recurse=False))
            print(f"{name:<25} | {params:<10,}")
            total_params += params
    print("-" * 40)
    print(f"{'Total Trainable':<25} | {total_params:<10,}")
