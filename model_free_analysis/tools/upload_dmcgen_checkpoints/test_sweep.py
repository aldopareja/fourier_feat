import pandas as pd
import numpy as np
from ml_logger import RUN
from pathlib import Path
from params_proto.neo_hyper import Sweep

from dmc_gen.config import Args
# loaded_sweep = Sweep(RUN, Args, Agent, Adapt, PreprocessArgs).load(pd.read_csv('hyperparams.csv'))
# envs = pd.read_csv('environments.csv').values.tolist()

envs = ['Cartpole-balance']

with Sweep(RUN, Args) as sweep:

    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # Args.eval_env = [f'distracting_control:{env_name}-hard-v1' for env_name in envs]  # eval-env does not matter

    with sweep.product:
        # Adapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        # NOTE: drqv2 directly uploads snapshot to the correct s3 path.
        Args.algorithm = ['soda', 'sac', 'pad', 'svea']
        Args.seed = [100]  # 10 random seeds for data-collection


@sweep.each
def tail(RUN, Args):
    env_name = Args.env_name.split(':')[-1][:-3].lower()
    if Args.algorithm == 'drqv2':
        # NOTE: only env_name==cartpole-balance-v1
        Args.snapshot_prefix = f'model-free/model-free/baselines/drqv2_original/train/test---{env_name}-v1/{Args.seed}'
    elif Args.algorithm in ['soda', 'sac', 'pad', 'svea']:
        # NOTE: only env_name==cartpole-balance-v1
        Args.snapshot_prefix = f'model-free/model-free/baselines/dmc_gen/train/test---dmc_gen_nockpt/{Args.algorithm}/{env_name}-v1/{Args.seed}'
    else:
        raise ValueError(f'algorithm {Args.algorithm} is invalid')

    # RUN.job_name = f"test---{RUN.now:%H.%M.%S}/{Args.algorithm}/{get_buffer_prefix(latent_buffer=True)}"
    RUN.job_name = f"test---{RUN.now:%H.%M.%S}/{Args.algorithm}/{env_name}/{Args.seed}"


sweep.save("test_sweep.jsonl")
