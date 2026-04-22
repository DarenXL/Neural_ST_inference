import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class CNN_ST_network(nn.Module):
    def __init__(self, in_channels=2, hidden=32, out_dim=4, spatial_dim=1, pool_option="max", nls=False, nls_sim=False, label=None):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.nls = nls
        self.nls_sim = nls_sim

        if label is None:
            self.label = f"{pool_option} {spatial_dim}D-Space CNN" + (" nls" if nls else "")
        else:
            self.label = label

        if spatial_dim == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            if pool_option == "avg":
                self.pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.pool = nn.AdaptiveMaxPool2d(1)

        elif spatial_dim == 2:
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, hidden, kernel_size=(3, 5, 5), padding=(1, 2, 2)
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
            nn.Linear(hidden + (out_dim * self.nls) if self.nls_sim == False else 2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, deepset=None, theta=None):

        B = x.shape[0]
        T = x.shape[2]

        summary = self.conv(x)
        summary = self.pool(summary)
        summary = summary.view(B, -1)

        if not self.nls:
            summary = global_mean_pool(summary, deepset)
        else:
            if not self.nls_sim :
                summary = torch.cat((summary, theta.float()), dim=1)
            else :
                theta_summary = self.conv(theta.float())
                # print(theta_summary.shape)
                theta_summary = self.pool(theta_summary)
                # print(theta_summary.shape)
                theta_summary = theta_summary.view(theta_summary.shape[0], -1)
                # print(theta_summary.shape)
                theta_summary = global_mean_pool(theta_summary, deepset)
                # print(theta_summary.shape, deepset.shape)
                # print(summary.shape, theta_summary.shape)
                summary = torch.cat((summary, theta_summary), dim=1)
                # print()

        out = self.dense(summary)
        return out

# class CNN_ST_network(nn.Module):
#     def __init__(self, in_channels=2, hidden=32, out_dim=4, spatial_dim=1, option="jointly", pool_option="avg", nls=False, nls_sim=False, label=None):
#         super().__init__()
#
#         self.option_type = option
#         self.spatial_dim = spatial_dim
#         self.nls = nls
#         self.nls_sim = nls_sim
#
#         if label is None:
#             self.label = f"{option} {pool_option} {spatial_dim}D-Space CNN" + (" nls" if nls else "")
#         else:
#             self.label = label
#
#         if option == "jointly":
#             if spatial_dim == 1:
#                 self.conv = nn.Sequential(
#                     nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2),
#                     nn.ReLU(),
#                     nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
#                     nn.ReLU(),
#                 )
#                 if pool_option == "avg" :
#                     self.pool = nn.AdaptiveAvgPool2d(1)
#                 else :
#                     self.pool = nn.AdaptiveMaxPool2d(1)
#
#             elif spatial_dim == 2:
#                 self.conv = nn.Sequential(
#                     nn.Conv3d(in_channels, hidden, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
#                     nn.ReLU(),
#                     nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
#                     nn.ReLU(),
#                 )
#                 if pool_option == "avg" :
#                     self.pool = nn.AdaptiveAvgPool3d(1)
#                 else :
#                     self.pool = nn.AdaptiveMaxPool3d(1)
#
#         elif option == "separately":
#
#             if spatial_dim == 1:
#                 self.spatial_conv = nn.Sequential(
#                     nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
#                     nn.ReLU(),
#                     nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
#                     nn.ReLU(),
#                 )
#                 if pool_option == "avg" :
#                     self.spatial_pool = nn.AdaptiveAvgPool1d(1)
#                 else :
#                     self.spatial_pool = nn.AdaptiveMaxPool1d(1)
#
#             elif spatial_dim == 2:
#                 self.spatial_conv = nn.Sequential(
#                     nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2),
#                     nn.ReLU(),
#                     nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
#                     nn.ReLU(),
#                 )
#                 if pool_option == "avg" :
#                     self.spatial_pool = nn.AdaptiveAvgPool2d(1)
#                 else :
#                     self.spatial_pool = nn.AdaptiveMaxPool2d(1)
#
#             self.temporal_conv = nn.Sequential(
#                 nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
#                 nn.ReLU(),
#             )
#             # if pool_option == "avg" :
#             #     self.temporal_pool = nn.AdaptiveAvgPool1d(1)
#             # else :
#             #     self.temporal_pool = nn.AdaptiveMaxPool1d(1)
#
#         self.dense = nn.Sequential(
#             nn.Linear(hidden + (out_dim * self.nls) if self.nls_sim == False else 2*hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, out_dim)
#         )
#
#     def forward(self, x, deepset=None, theta=None):
#
#         B = x.shape[0]
#         T = x.shape[2]
#
#         # print(deepset)
#
#         if self.option_type == "jointly":
#             summary = self.conv(x)
#             summary = self.pool(summary)
#             summary = summary.view(B, -1)
#
#         elif self.option_type == "separately":
#
#             if self.spatial_dim == 1:
#                 x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, -1, x.shape[-1])
#             else:
#                 x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, -1, x.shape[-2], x.shape[-1])
#
#             spatial_feat = self.spatial_conv(x_reshaped)
#             spatial_feat = self.spatial_pool(spatial_feat)
#
#             time_input = spatial_feat.view(B, T, -1).permute(0, 2, 1)
#
#             time_feat = self.temporal_conv(time_input)
#
#             summary = time_feat.mean(dim=2)
#
#         if not self.nls:
#             summary = global_mean_pool(summary, deepset)
#         else:
#             if not self.nls_sim :
#                 summary = torch.cat((summary, theta.float()), dim=1)
#             else :
#                 theta_summary = self.conv(theta.float())
#                 # print(theta_summary.shape)
#                 theta_summary = self.pool(theta_summary)
#                 # print(theta_summary.shape)
#                 theta_summary = theta_summary.view(theta_summary.shape[0], -1)
#                 # print(theta_summary.shape)
#                 theta_summary = global_mean_pool(theta_summary, deepset)
#                 # print(theta_summary.shape, deepset.shape)
#                 # print(summary.shape, theta_summary.shape)
#                 summary = torch.cat((summary, theta_summary), dim=1)
#                 # print()
#
#         out = self.dense(summary)
#         return out

# def print_parameters_per_layer(model):
#     print(f"{'Layer Name':<25} | {'Parameters':<10}")
#     print("-" * 40)
#     total_params = 0
#     for name, module in model.named_modules():
#         if len(list(module.parameters(recurse=False))) > 0:
#             params = sum(p.numel() for p in module.parameters(recurse=False))
#             print(f"{name:<25} | {params:<10,}")
#             total_params += params
#     print("-" * 40)
#     print(f"{'Total Trainable':<25} | {total_params:<10,}")
