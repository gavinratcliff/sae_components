import saeco.core as cl

from cross_layer_transcoder import CrossLayerTranscoder, CrossLayerTranscoderConfig
from saeco.components.features.optim_reset import OptimResetValuesConfig
from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig
from saeco.data.data_cfg import DataConfig
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.model_cfg import ActsDataConfig, ModelConfig
from saeco.data.split_config import SplitConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.trainer.normalizers import Aggregation, SAggregation
from saeco.trainer.run_config import RunConfig
from saeco.trainer.schedule_cfg import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "cross_layer_transcoder"

from saeco.mlog import mlog

mlog.init()


def acts_modifier(cache, acts):
    return acts + 1


cache = cl.Cache()
cache.register_write_callback("acts", acts_modifier)

dict_mult = 8
n_sites = 12

input_mlp_sites = ["transformer.h.{}.mlp.input".format(i) for i in range(n_sites)]

output_mlp_sites = ["transformer.h.{}.mlp.output".format(i) for i in range(n_sites)]

mlp_sites = input_mlp_sites + output_mlp_sites

data_config = DataConfig(
    dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            sites=mlp_sites,
            d_data=768,
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="bfloat16",
            storage_dtype_str="bfloat16",
        ),
        model_name="gpt2",
    ),
    trainsplit=SplitConfig(start=0, end=50, tokens_from_split=2_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**15,
        meta_batch_size=2**17,
        llm_batch_size=2**14,
    ),
    seq_len=256,
)

cfg = RunConfig[CrossLayerTranscoderConfig](
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
        input_sites=input_mlp_sites,
        target_sites=output_mlp_sites,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768 * n_sites, dict_mult=dict_mult),
    arch_cfg=CrossLayerTranscoderConfig(pre_bias=True, n_sites=n_sites),
)

# Disable std_E, set it to Aggregation.DONTUSE
cfg.normalizer_cfg.std_e = Aggregation.DONTUSE

g = CrossLayerTranscoder(cfg)
g.run_training()
