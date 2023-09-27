# CathSim: An Open-source Simulator for Endovascular Intervention
<!-- ### [[Project Page](https://robotvisionlabs.github.io/cathsim/)] [[Paper](https://arxiv.org/abs/2208.01455)] -->

<p align="center">
  <img src="./misc/cathsim_dn.gif" alt="animated" />
</p>

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Training](#training)
5. [Manual Control](#manual-control)
6. [Mesh Processing](#mesh-processing)


## Requirements
1. Ubuntu (tested with Ubuntu 22.04 LTS)
2. Miniconda (tested with 23.5, but all versions should work)
3. Python 3.9

If `miniconda` is not installed run the following for a quick Installation. Note: the script assumes you use `bash`.

```bash
# installing miniconda
mkdir -p ~/.miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
rm -rf ~/.miniconda3/miniconda.sh
~/.miniconda3/bin/conda init bash
source .bashrc
```

## Installation

1. Create a `conda environment`:

```bash
conda create -n cathsim python=3.9
conda activate cathsim
```

2. Install the environment:

<!-- git clone git@github.com:robotvision-ai/cathsim -->
```bash
# clone the repository (has to be manually for anonymous submission)
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the environment run with gym is to make use of the `make_dm_env` function and then wrap the resulting environment into a `DMEnvToGymWrapper` resulting in a `gym.Env`.

```python
from cathsim.utils import make_dm_env
from cathsim.wrappers import DMEnvToGymWrapper

env = make_dm_env(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom="phantom3",
    target="bca",
)

env = DMEnvToGymWrapper(env)

obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    for obs_key in obs:
        print(obs_key, obs[obs_key].shape)
    print(reward)
    print(done)
    for info_key in info:
        print(info_key, info[info_key])
```

For a list of the environment libraries at the current time, see the accompanying `environment.yml`

## Training 

In order to train the models available run:
```bash
bash ./scripts/train.sh
```

The script will create a `results` directory on the `cwd`. The script saves the data in `<trial-name>/<phantom>/<target>/<model>` format. Each model has three subfolders `eval`, `models` and `logs`, where the evaluation data contains the `Trajectory` data resulting from the evaluation of the policy, the `models` contains the `pytorch` models and the `logs` contains the `tensorboard` logs.

## Manual Control

For a quick visualization of the environment run:
```bash
run_env
```
You will now see the guidewire and the aorta along with the two sites that represent the targets. You can interact with the environment using the keyboard arrows.



## Terms of Use

Please review our [Terms of Use](TERMS.md) before using this project.

## License

Please feel free to copy, distribute, display, perform or remix our work but for non-commercial purposes only.
