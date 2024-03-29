if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from model_free_analysis.baselines import RUN
    import jaynes
    from sac_dennis_rff.sac import train
    from sac_dennis_rff.config import Args, Actor, Critic, Agent
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("mlp.jsonl")
    for kwargs in sweep:
        jaynes.config('supercloud')
        thunk = instr(train, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
