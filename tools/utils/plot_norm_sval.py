import csv
import matplotlib.pyplot as plt
from cmx import doc
import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def get_s_val(zs):
    # gram_matrix = zs @ zs.T
    sgv = torch.linalg.svdvals(zs)
    sgv /= sgv.sum()
    return sgv.cpu().numpy()

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
        s_vals = get_s_val(feats)

    return s_vals


if __name__ == '__main__':
    from ml_logger import logger
    import pandas as pd

    mean_std = pd.read_csv('drqv2_obs_norm_mean_std.csv')
    envs = mean_std['env_name'].tolist()
    b_vals = mean_std['b_val'].tolist()

    all_seeds = [100, 200, 300, 400, 500]
    colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']
    prefix = 'gs://ge-data-improbable/checkpoints/model-free/model-free/rff_post_iclr/dmc/drqv2/4_layer'

    with doc:
        for b_val, env_name in tqdm(zip(b_vals, envs), desc="b_val, env"):
            r = doc.table().figure_row()
            print(env_name)

            for seed in all_seeds:
                ddpg_agent_path = f"{prefix}/mlp/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/agent.pkl"
                ddpg_rb_path = f"{prefix}/mlp/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/replay_buffer.pkl"
                rfac_agent_path = f"{prefix}/rff_mean_std_full/rff/iso/b-{b_val}/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/agent.pkl"
                rfac_rb_path = f"{prefix}/rff_mean_std_full/rff/iso/b-{b_val}/{env_name.split(':')[-1][:-3]}/{seed}/checkpoint/replay_buffer.pkl"

                print(f"DDPG, env-{env_name}, seed-{seed}")
                ddpg_svals = calc_rank_from_rb(ddpg_agent_path, ddpg_rb_path)
                ddpg_svals = ddpg_svals[:50]

                print(f"RFAC, env-{env_name}, b-{b_val}, seed-{seed}")
                rfac_svals = calc_rank_from_rb(rfac_agent_path, rfac_rb_path)
                rfac_svals = rfac_svals[:50]

                width = 0.3
                x_range = list(range(50))

                plt.title(f"{env_name.split(':')[-1][:-3]}-{seed}", fontsize=18)
                plt.bar(x_range, ddpg_svals, color='black', label='DDPG')
                plt.bar([x + width for x in x_range], rfac_svals, color=colors[0], label='RFAC')
                plt.legend()
                plt.tight_layout()
                r.savefig(f"s_val_plot/seed_{seed}/{env_name.split(':')[-1][:-3]}_norm.png")
                plt.close()
    doc.flush()