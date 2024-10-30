from typing import Dict

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        mlp_layers: list = [256, 128],
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(
                    subspace,
                    features_dim=cnn_output_dim,
                    normalized_image=normalized_image,
                )
                total_concat_size += cnn_output_dim
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
