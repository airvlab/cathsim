from typing import Dict

from gymnasium import spaces

import torch.nn as nn
import torch
import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim

import gymnasium as gym

from x_transformers import Encoder, ViTransformerWrapper


class ViTExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        vit_output_dim: int = 256,
        normalized_image: bool = False,
        mlp_layers: list = [256, 128],
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = ViT(
                    subspace,
                    features_dim=vit_output_dim,
                    normalized_image=normalized_image,
                )
                total_concat_size += vit_output_dim
            else:
                extractors[key] = nn.Sequential()
                init_dim = get_flattened_obs_dim(subspace)
                for layer in mlp_layers:
                    extractors[key].add_module(
                        f"layer_{len(extractors[key])}", nn.Linear(init_dim, layer)
                    )
                    extractors[key].add_module(name="relu", module=nn.ReLU())
                    init_dim = layer

                total_concat_size += init_dim

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class ViT(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        patch_size: int = 10,
        normalized_image: bool = False,
        depth: int = 3,
        heads: int = 4,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ViT must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use ViT "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.vit = ViTransformerWrapper(
            image_size=observation_space.shape[1],
            patch_size=patch_size,
            channels=n_input_channels,
            attn_layers=Encoder(
                dim=features_dim,
                depth=depth,
                heads=heads,
            )
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.vit(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.vit(observations))


if __name__ == "__main__":

    encoder = ViTransformerWrapper(
        image_size=80,
        patch_size=10,
        channels=1,
        attn_layers=Encoder(
            dim=256,
            depth=3,
            heads=4
        )
    )

    img = torch.randn(1, 1, 80, 80)
    preds = encoder(img)
    print(preds.shape)
