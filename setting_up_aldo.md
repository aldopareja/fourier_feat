# Aldo's deployment log

Recording of all steps taken to set up a pycharm environment with remote deployment in a server.


## initial steps

- download this repo on both the server and locally
- set up sftp in `deployment` section of pycharm configurations
  - set up the ssh server and the remote directory to map the repo to
- set up a conda environment in the server with python 3.7 or bellow (for jaynes)
- set up an ssh interpreter in pycharm based on the sftp configuration

## setting up

- install jaynes both locally and in the ssh server.

```shell
conda activate base
pip install jaynes
```

### install mujoco on the server 

these come from instructions in the [original mujoco repo](https://github.com/openai/mujoco-py), this was done on an ubuntu server using `fish` as shell

```shell
cd ~
mkdir .mujoco
cd .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
pip install -U 'mujoco-py<2.2,>=2.1'
```

- install required packages
```shell
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt install libglew-dev
sudo apt install patchelf
sudo apt install mesa-utils
```

- add environment variables to the path
```shell
set -xg LD_LIBRARY_PATH $LD_LIBRARY_PATH /home/aldo/.mujoco/mujoco210/bin
set -xg LD_LIBRARY_PATH $LD_LIBRARY_PATH /usr/lib/nvidia
```

- check mujoco
```shell
$ pip3 install -U 'mujoco-py<2.2,>=2.1'
$ python3
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]
```

### install ml_logger

based on [ml_logger](https://github.com/geyang/ml_logger) - the instructions in the repo seem outdated.

```shell
sudo apt install libcurl4-gnutls-dev librtmp-dev
pip install ml-logger==0.8.72
```

### running the code

examples are in `https://github.com/Improbable-AI/fourier_feat/tree/main/model_free_analysis/sac_dennis_rff/dmc`

changing line `model_free_analysis/sac_dennis_rff/dmc/2_layer/train.py:9` to load `debug.jsonl` and not `lff.jsonl`

#### running notes

- I'm having issues with loading some yaml, I will comment line `model_free_analysis/__init__.py:22`
  so that I can see where the code breaks.
- error:
  ```shell
    File "/home/aldo/fourier_feat/model_free_analysis/sac_dennis_rff/dmc/2_layer/train.py", line 3, in <module>
      from model_free_analysis.baselines import RUN
  ModuleNotFoundError: No module named 'model_free_analysis.baselines'
  ```
  will change to import `RUN` directly from model_free_analysis.
- error:
  ```shell
  File "/home/aldo/fourier_feat/model_free_analysis/sac_dennis_rff/dmc/2_layer/train.py", line 9, in <module>
    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("debug.json")
  File "/home/aldo/miniconda3/envs/mit-rl/lib/python3.7/site-packages/params_proto/neo_hyper.py", line 324, in load
    sweep = self.read(sweep)
  File "/home/aldo/miniconda3/envs/mit-rl/lib/python3.7/site-packages/params_proto/neo_hyper.py", line 283, in read
    with open(filename, 'r') as f:
  FileNotFoundError: [Errno 2] No such file or directory: 'debug.json'
  ```

  This file is there but it's inside the same `2_layer` folder but I'm launching the script as a module. Otherwise the imports fail.
  
  To solve this I'll just append the currents file path to the name of the `.jsonl`.

- error:

  ```shell
  Exception: during comprehension of <ip: {env.JYNS_SLURM_HOST}>: 'types.SimpleNamespace' object has no attribute 'JYNS_SLURM_HOST'
  ```

  I can't config jaynes because there are some environment variables that need to be set. 

  I'll use `"local"` for the jaynes config.
  
  running:
  ```shell
  export JYNS_SLURM_HOST=127.0.0.1
  export JYNS_USERNAME=aldo
  export JYNS_SLURM_PEM=''
  export JYNS_PASSWORD=''
  export JYNS_AWS_INSTANCE_PROFILE=''
  export JYNS_GCP_PROJECT=''
  export JYNS_SLURM_DIR=''
  export JYNS_AWS_S3_BUCKET=''
  export JYNS_AWS_S3_BUCKET=''
  export JYNS_GS_BUCKET=''
  ```
  ```fish
  set -xg JYNS_SLURM_HOST 127.0.0.1
  set -xg JYNS_USERNAME aldo
  set -xg JYNS_SLURM_PEM ''
  set -xg JYNS_PASSWORD ''
  set -xg JYNS_AWS_INSTANCE_PROFILE ''
  set -xg JYNS_GCP_PROJECT ''
  set -xg JYNS_SLURM_DIR ''
  set -xg JYNS_AWS_S3_BUCKET ''
  set -xg JYNS_AWS_S3_BUCKET ''
  set -xg JYNS_GS_BUCKET ''
  ```
- error: 
  ```shell
  UnicodeEncodeError: 'latin-1' codec can't encode character '\u2713' in position 5: ordinal not in range(256)
  ```
  
  solution:
  ```shell
  export LANG=en_US.UTF-8
  ```
  
- error on glfw not being initialized:

  solution:
  ```shell
  export MUJOCO_GL=egl
  ```

- installed packages:
  ```shell
  torch
  gym==0.21
  tqdm
  pandas
  dm_control
  gym-dmc
  ```