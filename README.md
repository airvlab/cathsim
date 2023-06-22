# CathSim: An Open-source Simulator for Endovascular Intervention
### *[Tudor Jianu](https://tudorjnu.github.io/), [Baoru Huang](https://baoru.netlify.app), [Mohamed E. M. K. Abdelaziz](https://memkabdelaziz.com/), [Minh Nhat Vu](https://www.acin.tuwien.ac.at/staff/mnv/), [Sebastiano Fichera](https://www.liverpool.ac.uk/engineering/staff/sebastiano-fichera/), [Chun-Yi Lee](https://elsalab.ai/about), [Pierre Berthet-Rayne](https://caranx-medical.com/pierre-berthet-rayne-phd-ing/), [Ferdinando Rodriguez y Baena](https://www.imperial.ac.uk/people/f.rodriguez), [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)*
### [[Project Page](https://robotvision-ai.github.io/cathsim/)] [[Paper](https://arxiv.org/abs/2208.01455)]


![CathSim](./cathsim.png)

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Training](#training)


## Requirements
1. Add enviromnent Requirements (Ubuntu, etc.)
2. Any other Requirements


## Installation

1. Create a `conda environment`:

```bash
conda create -n cathsim python=3.9
conda activate cathsim
```

2. Install the environment:

```bash
git clone -b git@github.com:cathsim/cathsim.github.io.git
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the enviromnent run with gym is to make use of the `make_dm_env` function and then wrap the resulting environment into a `DMEnvToGymWrapper` resulting in a `gym.Env`.

```python
from cathsim.cathsim.env_utils import make_dm_env
from cathsim.wrappers.wrappers import DMEnvToGymWrapper

env = make_dm_env(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom='phantom3',
    target='bca',
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

## Training 

In order to train the modells available run:
```bash
bash ./scripts/train.sh
```

## License
Creative Commons
