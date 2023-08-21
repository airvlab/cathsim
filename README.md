# CathSim: An Open-source Simulator for Endovascular Intervention
### [[Project Page](https://robotvisionlabs.github.io/cathsim/)] [[Paper](https://arxiv.org/abs/2208.01455)]


<div align="center">
    <a href="https://"><img height="auto" src="/misc/cathsim_dn.gif"></a>
</div>

![CathSim](./cathsim.png)

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Training](#training)
5. [Manual Control](#manual-control)
6. [Mesh Processing](#mesh-processing)


## Requirements
1. Ubuntu (tested with Ubuntu 22.04 LTS)
2. Miniconda (tested with 23.5 but all versions should work)
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

```bash
git clone git@github.com:robotvision-ai/cathsim
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the enviromnent run with gym is to make use of the `make_dm_env` function and then wrap the resulting environment into a `DMEnvToGymWrapper` resulting in a `gym.Env`.

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

The script will create a results directory on the `cwd`. The script saves the data in `<trial-name>/<phantom>/<target>/<model>` format. Each model has three subfolders `eval`, `models` and `logs`, where the evaluation data contains the `Trajectory` data resulting from the evaluation of the policy, the `models` contains the `pytorch` models and the `logs` contains the `tensorboard` logs.

## Manual Control

For a quick visualisation of the environment run:
```bash
run_env
```
You will now see the guidewire and the aorta along with the two sites that represent the targets. You can interact with the environment using the keyboard arrows.

## Mesh Processing

You can use a custom aorta by making use of V-HACD convex decomposition. To do so, you can use stl2mjcf, available [here](https://github.com/tudorjnu/stl2mjcf). You can quickly install the tool with:

```bash
pip install git+git@github.com:tudorjnu/stl2mjcf.git
```

After the installation, you can use `stl2mjcf --help` to see the available commands. The resultant files can be then added to `cathsim/assets`. The `xml` will go in that folder and the resultant meshes folder will go in `cathsim/assets/meshes/`. 

Note: You will probably have to change the parameters of V-HACD for the best results.

## TODO's

- [x] Code refactoring
- [x] Add fluid simulation
- [x] Add VR/AR interface through Unity
- [x] Implement multiple aortic models
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

## Terms of Use

Please review our [Terms of Use](TERMS.md) before using this project.

## License

Please feel free to copy, distribute, display, perform or remix our work but for non-commercial porposes only.

## Citation

If you find our paper useful in your research, please consider citing:

``` bibtex
@article{jianu2022cathsim,
  title={CathSim: An Open-source Simulator for Endovascular Intervention},
  author={Jianu, Tudor and Huang, Baoru and Abdelaziz, Mohamed EMK and Vu, Minh Nhat and Fichera, Sebastiano and Lee, Chun-Yi and Berthet-Rayne, Pierre and Nguyen, Anh and others},
  journal={arXiv preprint arXiv:2208.01455},
  year={2022}
