import csv

import numpy as np
import torch
from tqdm import tqdm


def H_d(ps):
    ps_norm = ps / ps.sum()
    vec = np.log(ps) * ps_norm
    return - np.sum(vec)


@torch.no_grad()
def get_effective_rank(zs):
    gram_matrix = zs @ zs.T
    sgv = torch.linalg.svdvals(gram_matrix)
    sgv /= sgv.sum()
    return np.exp(H_d(sgv.cpu().numpy() + 1e-6))

@torch.no_grad()
def get_nuclear_norm(zs):
    sgv = torch.linalg.svdvals(zs)
    return sgv.sum().cpu().numpy()

@torch.no_grad()
def get_srank(zs):
    sgv = torch.linalg.svdvals(zs)
    sgv /= sgv.sum()
    sgv = sgv.cpu().numpy().flatten()
    srank = sgv.shape[0]
    total_sum = 1.0
    while total_sum >= 0.99:
        total_sum -= sgv[srank-1]
        srank -= 1
    srank += 1
    return srank


def calc_rank_from_rb(agent_path, rb_path):
    from ml_logger import logger
    rb = logger.torch_load(rb_path, map_location='cpu')
    rb.obses = rb.obses[~np.any(np.isnan(rb.actions), axis=1)]
    rb.actions = rb.actions[~np.any(np.isnan(rb.actions), axis=1)]
    rb.obses = rb.obses[~np.any((np.abs(rb.actions)>1), axis=1)]
    rb.actions = rb.actions[~np.any((np.abs(rb.actions)>1), axis=1)]

    agent = logger.torch_load(agent_path, map_location='cpu')
    inds = np.arange(rb.obses.shape[0])
    np.random.shuffle(inds)
    obses = rb.obses[inds[:1000]]
    acts = rb.actions[inds[:1000]]
    obs_tensor = torch.as_tensor(obses).float()
    action_tensor = torch.as_tensor(acts).float()

    (_, feats), _ = agent.critic(obs_tensor, action_tensor, get_feat=True)

    with logger.time("effective rank"):
        rank = get_srank(feats)

    logger.print(f'eval/q_rank: {rank}')
    return rank


if __name__ == '__main__':
    from ml_logger import logger
    import pandas as pd

    mean_std = pd.read_csv('drqv2_obs_norm_mean_std.csv')
    envs = mean_std['env_name'].tolist()
    b_vals = mean_std['b_val'].tolist()

    all_seeds = [100, 200, 300, 400, 500]
    prefix = 'gs://ge-data-improbable/checkpoints/model-free/model-free/rff_post_iclr/dmc/drqv2/4_layer'

    with open('drqv2_srank_table.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['env_name', 'DDPG (Mean)', 'DDPG (SE)', 'RFAC (Mean)', 'RFAC (SE)'])
        for b_val, env_name in tqdm(zip(b_vals, envs), desc="b_val, env"):
            print(env_name)
            ddpg_rank_lst = []
            rfac_rank_lst = []

            for seed in all_seeds:
                ddpg_agent_path = f"{prefix}/mlp/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/agent.pkl"
                ddpg_rb_path = f"{prefix}/mlp/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/replay_buffer.pkl"
                rfac_agent_path = f"{prefix}/rff_mean_std_full/rff/iso/b-{b_val}/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/agent.pkl"
                rfac_rb_path = f"{prefix}/rff_mean_std_full/rff/iso/b-{b_val}/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/replay_buffer.pkl"

                print(f"DDPG, env-{env_name}, seed-{seed}")
                ddpg_rank = calc_rank_from_rb(ddpg_agent_path, ddpg_rb_path)
                ddpg_rank_lst.append(ddpg_rank)

                print(f"RFAC, env-{env_name}, b-{b_val}, seed-{seed}")
                rfac_rank = calc_rank_from_rb(rfac_agent_path, rfac_rb_path)
                rfac_rank_lst.append(rfac_rank)

            writer.writerow(
                [env_name.split(':')[-1][:-3], np.mean(ddpg_rank_lst), np.std(ddpg_rank_lst), np.mean(rfac_rank_lst),
                 np.std(rfac_rank_lst)])
