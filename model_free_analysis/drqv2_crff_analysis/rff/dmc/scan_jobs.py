
if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from model_free_analysis.baselines import RUN
    import jaynes
    from drqv2_crff.drqv2_crff import train
    from drqv2_crff.config import Args, Agent
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Args, Agent).load("rff.jsonl")

    for kwargs in sweep:
        if needs_relaunch(RUN.prefix, stale_limit=30, not_exist_ok=True):
            pass

    jaynes.listen()
