import pandas as pd
import numpy as np
from ml_logger import RUN
from pathlib import Path
from params_proto.neo_hyper import Sweep

from dmc_gen.config import Args
# loaded_sweep = Sweep(RUN, Args, Agent, Args, PreprocessArgs).load(pd.read_csv('hyperparams.csv'))
# envs = pd.read_csv('environments.csv').values.tolist()

soda_envs = ['Walker-walk', 'Walker-stand', 'Cartpole-swingup', 'Ball_in_cup-catch', 'Finger-spin']
extra_envs = ['Reacher-easy', 'Cheetah-run', 'Cartpole-balance', 'Hopper-hop']

envs = soda_envs + extra_envs

with Sweep(RUN, Args) as sweep:

    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # Args.eval_env = [f'distracting_control:{env_name}-hard-v1' for env_name in envs]  # eval-env does not matter

    with sweep.product:
        # Args.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        # NOTE: drqv2 directly uploads snapshot to the correct s3 path.
        Args.algorithm = ['soda', 'sac', 'pad', 'svea']
        Args.seed = [(i + 1) * 100 for i in range(5)]  # 10 random seeds for data-collection


@sweep.each
def tail(RUN, Args):
    env_name = Args.train_env.split(':')[-1][:-3].lower()
    if Args.algorithm == 'drqv2':
        Args.snapshot_prefix = f'model-free/model-free/baselines/drqv2_original/train/{env_name}-v1/{Args.seed}'
    elif Args.algorithm in ['soda', 'sac', 'pad', 'svea']:
        Args.snapshot_prefix = f'model-free/model-free/baselines/dmc_gen/train/dmc_gen_nockpt/{Args.algorithm}/{env_name}-v1/{Args.seed}'
    else:
        raise ValueError(f'algorithm {Args.algorithm} is invalid')

    RUN.job_name = f"{RUN.now:%H.%M.%S}/{Args.algorithm}/{env_name}/{Args.seed}"

sweep.save("sweep.jsonl")
