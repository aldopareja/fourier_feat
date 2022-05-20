if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from model_free_analysis.baselines import RUN
    import jaynes
    from sac_dennis_rff.sac import train
    from sac_dennis_rff.config import Args, Actor, Critic, Agent
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("debug.jsonl")
    # runner = None
    jaynes.config('supercloud')

    for i, kwargs in enumerate(sweep):
        thunk = instr(train, **kwargs)
        # if i % 2 == 0:
        #     runner = jaynes.add(thunk)
        # else:
        #     runner = runner.chain(thunk)
        #     runner.execute()
        jaynes.run(thunk)

    jaynes.listen()
