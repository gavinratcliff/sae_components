from functools import cached_property

import saeco.components.features.features as ft

import saeco.core as cl
import torch
import torch.nn as nn

from saeco.architecture import Architecture, aux_model_prop, loss_prop, model_prop, SAE

from saeco.components import L2Loss, Lambda, Loss, SparsityPenaltyLoss
from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.misc import useif
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


class MatryoshkaCLTConfig(SweepableConfig):
    n_sites: int = 12
    n_nestings: int = 3


class MatryoshkaLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y.unsqueeze(0) - y_pred) ** 2)


class MatryoshkaCLT(Architecture[MatryoshkaCLTConfig]):
    def setup(self):
        assert self.init.d_dict % self.cfg.n_sites == 0
        assert self.init.d_data % self.cfg.n_sites == 0
        self.d_layer_dict = self.init.d_dict // self.cfg.n_sites
        self.d_layer_data = self.init.d_data // self.cfg.n_sites

        assert (2 ** (self.cfg.n_nestings - 1)) <= self.d_layer_dict

        self.nesting_sizes = [
            self.d_layer_dict // (2**i) for i in range(self.cfg.n_nestings)
        ]

    def generate_cross_layer_decode(self, nesting_size):
        return cl.Parallel(
            *[
                Seq(
                    Lambda(
                        lambda x, idx=i: torch.cat(
                            [
                                x[
                                    :,
                                    self.d_layer_dict * j : self.d_layer_dict * j
                                    + nesting_size,
                                ]
                                for j in range(idx + 1)
                            ],
                            dim=1,
                        )
                    ),
                    nn.Linear(
                        in_features=nesting_size * (i + 1),
                        out_features=self.d_layer_data,
                    ),
                )
                for i in range(self.cfg.n_sites)
            ]
        ).reduce(lambda *x: torch.cat(x, dim=1))

    @cached_property
    def encode_each_layer(self):

        encode_module = cl.Parallel(
            *[
                Seq(
                    Lambda(
                        lambda x, idx=i: x[
                            :, self.d_layer_data * idx : self.d_layer_data * (idx + 1)
                        ]
                    ),
                    nn.Linear(
                        in_features=self.d_layer_data,
                        out_features=self.d_layer_dict,
                    ),
                )
                for i in range(self.cfg.n_sites)
            ]
        ).reduce(lambda *x: torch.cat(x, dim=1))

        return Seq(
            weight=encode_module,
            nonlinearity=nn.ReLU(),  # Whatever nonlinearity you want
        )

    @model_prop
    def model(self):
        return SAE(
            encoder=self.encode_each_layer,
            decoder=cl.Parallel(
                *[
                    self.generate_cross_layer_decode(nesting_size)
                    for nesting_size in self.nesting_sizes
                ]
            ).reduce(lambda *x: torch.stack(x, dim=0)),
        )

    @loss_prop
    def l2_loss(self):
        return MatryoshkaLoss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
