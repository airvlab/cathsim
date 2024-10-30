import pprint
from pathlib import Path

import torch as th
import yaml
from mergedeep import merge

RESULTS_PATH = Path.cwd() / "results"


class Config:
    def __init__(self, **kwargs):
        """Config class for managing hyperparameters."""
        self.set_defaults()
        self.update(kwargs)
        if self.config_name != "default":
            self.load(self.config_name)

    def set_defaults(self):
        """Set default values for the config."""
        from cathsim.rl.feature_extractors import CustomExtractor

        self.base_path = RESULTS_PATH
        self.trial_name = "test"
        self.config_name = "default"

        self.task_kwargs = dict(
            image_size=80,
            phantom="phantom3",
            target="bca",
        )
        self.task_kwargs = dict(
            use_pixels=True,
            use_segment=False,
            image_size=80,
            phantom="phantom3",
            target="bca",
            random_init_distance=1e-3,
            sample_target=False,
            target_from_sites=False,
        )

        self.wrapper_kwargs = dict(
            time_limit=300,
            grayscale=True,
            channels_first=False,
            use_obs=["pixels", "guidewire", "joint_pos", "joint_vel"],
        )

        self.algo_kwargs = dict(
            buffer_size=int(5e5),
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=CustomExtractor,
            ),
            device="cuda" if th.cuda.is_available() else "cpu",
        )

    def update(self, overrides: dict):
        """Update the config with new values.

        This function is used to update the config with new values.
        It is used to update the config with the values from the YAML file using nested dictionaries.

        Args:
            overrides (dict): Config values to override
        """
        self.__dict__ = merge(self.__dict__, overrides)

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __add__(self, other):
        new_config = Config()
        new_config.__dict__ = merge(self.__dict__, other.__dict__)
        return new_config

    def load(self, config_name: str):
        """Load a config from a YAML file and update the config.

        Args:
            config_name (str): Name of the config to load
        """
        with open(Path(__file__).parent / "config" / f"{config_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.update(config)

    def get_env_path(self):
        phantom_name = self.task_kwargs["phantom"]
        target_name = self.task_kwargs["target"]
        return (
            self.base_path
            / self.trial_name
            / phantom_name
            / target_name
            / self.config_name
        )


if __name__ == "__main__":
    config = Config()
    print(config)

    config = Config(config_name="full")
    print(config)
