# CathSim: An Open-source Simulator for Endovascular Intervention
### [[Project Page](https://RobotVisionAI.github.io/cathsim/)] [[Paper](https://arxiv.org/abs/2208.01455)]


![CathSim](./mics/cathsim_dn.gif)

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Training](#training)
5. [Manual Control](#manual_control)



## Requirements
1. Ubuntu (tested with Ubuntu 22.04 LTS)
2. Miniconda 

If `miniconda` is not installed run the following for a quick Installation. Note: the script assumes you use `bash`.

```bash
# installing miniconda
mkdir -p ~/.miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
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

```bash
git clone git@github.com:robotvision-ai/cathsim
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

In order to train the models available run:
```bash
bash ./scripts/train.sh
```

The script will create a results directory on the `cwd`. The script saves the data in `<phantom>/<target>/<model>` format. Each model has three subfolders `eval`, `models` and `logs`, where the evaluation data contains the `np.array` data resulting from the evaluation of the policy, the `models` contains the `pytorch` models and the `logs` contains the `tensorboard` logs.


## Control
Describle how to control the catheter manually with the keyboard here.


## TODO's
- [x] Code refactoring
- [x] Add fluid simulation
- [x] Add VR/AR interface through Unity
- [ ] Implement multiple aortic models
- [ ] Add guidewire representation



## Contributors
- [Tudor Jianu](https://tudorjnu.github.io/)
- [Baoru Huang](https://baoru.netlify.app)
- Jingxuan Kang
- Tuan Van Vo
- [Mohamed E. M. K. Abdelaziz](https://memkabdelaziz.com/)
- [Minh Nhat Vu](https://www.acin.tuwien.ac.at/staff/mnv/)
- [Sebastiano Fichera](https://www.liverpool.ac.uk/engineering/staff/sebastiano-fichera/)
- [Chun-Yi Lee](https://elsalab.ai/about)
- [Olatunji Mumini Omisore](https://sites.google.com/view/moom1)
- [Pierre Berthet-Rayne](https://caranx-medical.com/pierre-berthet-rayne-phd-ing/)
- [Ferdinando Rodriguez y Baena](https://www.imperial.ac.uk/people/f.rodriguez)
- [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)


## License
Please feel free to copy, distribute, display, perform or remix our work but for non-commercial porposes only.
