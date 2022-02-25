from pathlib import Path
import pandas as pd
from params_proto.neo_hyper import Sweep

from drqv2_crff.config import Args, Agent
from model_free_analysis import RUN

hyperparams = pd.read_csv('pretrain_hyperparams.csv')
hyperparams = hyperparams[hyperparams['env_name'].isin(['dmc:Cheetah-run-v1',
                                                        'dmc:Hopper-hop-v1',
                                                        'dmc:Walker-walk-v1',
                                                        'dmc:Quadruped-walk-v1',
                                                        'dmc:Quadruped-run-v1',])]

with Sweep(RUN, Args, Agent) as sweep:
    Args.replay_buffer_num_workers = 3
    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.tmp_dir = "/tmp"

    with sweep.product:
        with sweep.zip:
            Args.env_name = hyperparams['env_name'].tolist()
            Args.train_frames = hyperparams['train_frames'].tolist()
            Args.replay_buffer_size = hyperparams['replay_buffer_size'].tolist()
            Args.batch_size = hyperparams['batch_size'].tolist()
            Args.nstep = hyperparams['nstep'].tolist()
            Agent.stddev_schedule = hyperparams['stddev_schedule'].tolist()
            Agent.lr = hyperparams['lr'].tolist()
            Agent.feature_dim = hyperparams['feature_dim'].tolist()

        with sweep.zip:
            Agent.wavelet_transform = [True, False, True]
            Agent.wavelet_only_low = [False, False, True]

        Args.seed = [100, 200, 300, 400, 500]

@sweep.each
def tail(RUN, Args, Agent):
    if Agent.wavelet_transform:
        if Agent.wavelet_only_low:
            suffix = 'wavelet_only_low'
        else:
            suffix = 'wavelet'
    else:
        suffix = 'mlp'

    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{suffix}/{Args.env_name.split(':')[-1][:-3]}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")
