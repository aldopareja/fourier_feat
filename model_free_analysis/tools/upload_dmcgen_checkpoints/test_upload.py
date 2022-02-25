if __name__ == '__main__':
    from ml_logger import instr
    from model_free_analysis import RUN
    import jaynes
    from dmc_gen.upload_checkpoints import main
    from dmc_gen.config import Args
    from params_proto.neo_hyper import Sweep

    # sweep = Sweep(RUN, Args, Agent, Adapt).load("sweep.jsonl")
    sweep = Sweep(RUN, Args).load("test_sweep.jsonl")

    jaynes.config("local" if RUN.debug else "tticbirch")
    for kwargs in sweep:
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
