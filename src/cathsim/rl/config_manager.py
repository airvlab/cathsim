import torch as th
import yaml
import pprint

from pathlib import Path
from mergedeep import merge

RESULTS_PATH = Path.cwd() / "results"


class Config:
    def __init__(self, **kwargs):
        self.set_defaults()
        self.update(kwargs)
        if self.config_name != "default":
            self.load(self.config_name)

    def set_defaults(self):
        from cathsim.rl.feature_extractors import CustomExtractor

        self.base_path = RESULTS_PATH
        self.trial_name = "test"
        self.config_name = "default"

        self.task_kwargs = {
            "image_size": 80,
            "phantom": "phantom3",
            "target": "bca",
        }
        self.task_kwargs = dict(
            use_pixels=True,
            use_segment=True,
            image_size=80,
            phantom="phantom3",
            target="bca",
            random_init_distance=1e-3,
            sample_target=False,
            target_from_sites=True,
        )

        self.wrapper_kwargs = {
            "time_limit": 300,
            "grayscale": True,
            "channels_first": False,
            "use_obs": ["pixels", "guidewire", "joint_pos", "joint_vel"],
        }

        self.algo_kwargs = {
            "buffer_size": int(5e5),
            "policy": "MultiInputPolicy",
            "policy_kwargs": {
                "features_extractor_class": CustomExtractor,
            },
            "device": "cuda" if th.cuda.is_available() else "cpu",
        }

    def update(self, overrides: dict):
        self.__dict__ = merge(self.__dict__, overrides)

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __add__(self, other):
        new_config = Config()
        new_config.__dict__ = merge(self.__dict__, other.__dict__)
        return new_config

    def load(self, config_name: str):
        with open(Path(__file__).parent / "config" / f"{config_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.update(config)

    def get_env_path(self):
        return (
            self.base_path
            / self.trial_name
            / f"{self.task_kwargs['phantom']}/{self.task_kwargs['target']}/{self.config_name}"
        )


if __name__ == "__main__":
    config = Config()
    print(config)

    config = Config(config_name="full")
    print(config)
