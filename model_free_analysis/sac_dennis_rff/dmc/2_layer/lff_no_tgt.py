from pathlib import Path

from params_proto.neo_hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from model_free_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.train_frames = 1_000_000
    Args.env_name = 'dmc:Quadruped-run-v1'
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Agent.use_rff = True
    Agent.learnable_temperature = True
    Agent.critic_tau = None

    Agent.actor_fourier_features = 3120
    Agent.critic_fourier_features = 3600

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        Agent.scale = [0.00001, 0.00003, 0.00007, 0.007, 0.01, 0.03] #[0.0001, 0.0003, 0.001, 0.003]

@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    if Agent.learnable_temperature:
        suffix = 'alpha_tune'
    else:
        suffix = f'alpha_fixed-{Agent.init_temperature}'

    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{suffix}/scale-{Agent.scale}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")