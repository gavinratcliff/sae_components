import math
from functools import cached_property

import saeco.components.features.features as ft

import saeco.core as cl
import torch
import torch.nn as nn

from saeco.architecture import Architecture, aux_model_prop, loss_prop, model_prop, SAE

from saeco.components import Lambda, Loss, SparsityPenaltyLoss
from saeco.components.features.optim_reset import OptimResetValuesConfig

from saeco.components.losses.losses import L2Loss
from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig

from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.data.data_cfg import DataConfig
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.model_cfg import ActsDataConfig, ModelConfig
from saeco.data.split_config import SplitConfig
from saeco.initializer.initializer_config import InitConfig

from saeco.misc import useif
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from saeco.trainer.normalizers import Aggregation, SAggregation
from saeco.trainer.run_config import RunConfig
from saeco.trainer.schedule_cfg import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "cross_layer_transcoder"

from saeco.mlog import mlog


class MatryoshkaConfig(SweepableConfig):
    pre_bias: bool = False  # placeholder until more config options are needed
    n_nestings: int = 2  # Number of nested layers


class MatryoshkaLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y.unsqueeze(0) - y_pred) ** 2)


class Matryoshka(Architecture[MatryoshkaConfig]):
    def setup(self):
        assert (2 ** (self.cfg.n_nestings - 1)) <= self.init.d_dict

        self.nesting_sizes = [
            self.init.d_dict // (2**i) for i in range(self.cfg.n_nestings)
        ]
        print("NESTING SIZES")
        print(self.nesting_sizes)

    @cached_property
    def encoder(self):
        return Seq(
            **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
            weight=self.init.encoder,
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),  # Whatever nonlinearity you want
        )

    @model_prop
    def model(self):

        return SAE(
            encoder=self.encoder,
            decoder=cl.Parallel(
                *[
                    Seq(
                        Lambda(lambda x, idx=nesting_size: x[:, :idx]),
                        nn.Linear(
                            in_features=nesting_size, out_features=self.init.d_data
                        ),
                    )
                    for nesting_size in self.nesting_sizes
                ]
            ).reduce(lambda *x: torch.stack(x, dim=0)),
        )

    # loss_prop designates a Loss that will be used in training
    @loss_prop
    def l2_loss(self):
        return MatryoshkaLoss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)


mlog.init()


def acts_modifier(cache, acts):
    return acts + 1


cache = cl.Cache()
cache.register_write_callback("acts", acts_modifier)

dict_mult = 8

sites = ["transformer.h.5.input"]

data_config = DataConfig(
    dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            sites=sites,
            d_data=768,
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="bfloat16",
            storage_dtype_str="bfloat16",
        ),
        model_name="gpt2",
    ),
    trainsplit=SplitConfig(start=0, end=50, tokens_from_split=500_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**15,
        meta_batch_size=2**17,
        llm_batch_size=2**14,
    ),
    seq_len=256,
)

cfg = RunConfig[MatryoshkaConfig](
    train_cfg=TrainConfig(
        data_cfg=data_config,
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=500,
            resample_period=8_000,
            lr_cooldown_length=0.5,
            lr_warmup_length=500,
        ),
        #
        batch_size=2048,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 1.1e-3,
            "l2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        intermittent_metric_freq=1000,
        input_sites=sites,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=dict_mult),
    arch_cfg=MatryoshkaConfig(pre_bias=True),
)

# Disable std_E, set it to Aggregation.DONTUSE
cfg.normalizer_cfg.std_e = Aggregation.DONTUSE

g = Matryoshka(cfg)
g.run_training()
