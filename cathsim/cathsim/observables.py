import yaml
import numpy as np
from pathlib import Path

from dm_control.composer.observation.observable import MujocoCamera
from dm_control.composer.observation import observable
from dm_control import composer
from dm_env import specs

config_path = Path(__file__).parent / 'env_config.yaml'
with open(config_path.as_posix()) as f:
    env_config = yaml.safe_load(f)


class CameraObservable(MujocoCamera):
    def __init__(self, camera_name, height=128, width=128, corruptor=None,
                 depth=False, preprocess=False, grayscale=False,
                 segmentation=False, scene_option=None):
        super().__init__(camera_name, height, width)
        self._dtype = np.uint8
        self._n_channels = 1 if segmentation else 3
        self._preprocess = preprocess
        self.scene_option = scene_option
        self.segmentation = segmentation

    def _callable(self, physics):
        def get_image():
            image = physics.render(  # pylint: disable=g-long-lambda
                self._height, self._width, self._camera_name, depth=self._depth,
                scene_option=self.scene_option, segmentation=self.segmentation)
            if self.segmentation:
                geom_ids = image[:, :, 0]
                if np.all(geom_ids == -1):
                    return np.zeros((self._height, self._width, 1), dtype=self._dtype)
                geom_ids = geom_ids.astype(np.float64) + 1
                geom_ids = geom_ids / geom_ids.max()
                image = 255 * geom_ids
                image = np.expand_dims(image, axis=-1)
            image = image.astype(self._dtype)
            return image
        return get_image

    @property
    def array_spec(self):
        return specs.BoundedArray(
            shape=(self._height, self._width, self._n_channels),
            dtype=self._dtype,
            minimum=0,
            maximum=255,
        )


class JointObservables(composer.Observables):

    @ composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @ composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)
