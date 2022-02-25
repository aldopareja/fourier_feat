import pickle
import csv
import numpy as np
import torch
from ml_logger import logger

path = 'gs://ge-data-improbable/checkpoints/model-free/model-free/rff_post_iclr/dmc/drq/4_layer/mlp/{env}/{seed}/checkpoint/replay_buffer.pkl'

envs = ['Acrobot-swingup', 'Quadruped-run', 'Quadruped-walk', 'Humanoid-run', 'Finger-turn_hard', 'Walker-run', 'Cheetah-run', 'Hopper-hop']
seeds = [100, 200, 300, 400, 500]

with open('/Users/aajay/Desktop/drq_obs_norm_mean_std.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['env_name', 'bias', 'scale'])

    for env in envs:
        obses = []
        for seed in seeds:
            curr_path = path.format(env=env, seed=seed)
            replay_buffer = logger.load_torch(curr_path)
            obses.append(replay_buffer.obses)
        obses = np.concatenate(obses, axis=0)

        writer.writerow([f'dmc:{env}-v1', obses.mean(axis=0).tolist(), obses.std(axis=0).tolist()])
