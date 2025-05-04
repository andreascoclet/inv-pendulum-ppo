# model.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import List, Tuple

class ActorCritic(nn.Module):
    """ Shared MLP backbone with Gaussian policy heads (mu & global log‑std). """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64)):
        super().__init__()
        # shared backbone
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for out_dim in hidden_sizes:
            layers += [ self.layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh() ]
            in_dim = out_dim
        self.base = nn.Sequential(*layers)
        # policy head
        mu_layers: List[nn.Module] = []
        head_in = in_dim
        for h in hidden_sizes[:-1]:
            mu_layers += [ self.layer_init(nn.Linear(head_in, h), std=0.01), nn.Tanh() ]
            head_in = h
        mu_layers.append(self.layer_init(nn.Linear(head_in, action_dim), std=0.01))
        self.mu_head = nn.Sequential(*mu_layers)
        # global log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # value head
        v_layers: List[nn.Module] = []
        head_in = in_dim
        for h in hidden_sizes[:-1]:
            v_layers += [ self.layer_init(nn.Linear(head_in, h), std=1.0) ]
            head_in = h
        v_layers.append(self.layer_init(nn.Linear(head_in, 1), std=1.0))
        self.v_head = nn.Sequential(*v_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.base(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    @staticmethod
    def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer