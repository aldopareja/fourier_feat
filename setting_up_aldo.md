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
```

- add environment variables to the path
```shell
set -xg LD_LIBRARY_PATH $LD_LIBRARY_PATH /home/aldo/.mujoco/mujoco210/bin
set -xg LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
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

based on [ml_logger](https://github.com/geyang/ml_logger)

```shell
sudo apt install libcurl4-gnutls-dev librtmp-dev
pip install ml-logger==0.7.11
```

### running the code

examples are in `https://github.com/Improbable-AI/fourier_feat/tree/main/model_free_analysis/sac_dennis_rff/dmc`

changing line `model_free_analysis/sac_dennis_rff/dmc/2_layer/train.py:9` to load `debug.jsonl` and not `lff.jsonl`
