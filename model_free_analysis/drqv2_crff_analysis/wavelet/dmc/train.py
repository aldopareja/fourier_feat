if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from model_free_analysis.baselines import RUN
    import jaynes
    from drqv2_crff.drqv2_crff import train
    from drqv2_crff.config import Args, Agent
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Args, Agent).load("mlp.jsonl")

    for kwargs in sweep:
        # with logger.Prefix(RUN.prefix):
        #     status = logger.read_params('job.status', default=None, not_exist_ok=True)
        #     if status == "running":
        #         logger.print(f"{RUN.prefix} is {status}", color='green')
        #     else:
        thunk = instr(train, **kwargs)
        jaynes.config("local" if RUN.debug else "gcp", launch=dict(name="aajay-" + RUN.prefix[-50:]))
        logger.job_requested(job=dict(instance_id=jaynes.run(thunk)))
    jaynes.listen()
