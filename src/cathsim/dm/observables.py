import numpy as np

from dm_control.composer.observation.observable import MujocoCamera
from dm_control.composer.observation import observable
from dm_control import composer
from dm_env import specs

from cathsim.dm.utils import get_env_config


env_config = get_env_config()


class CameraObservable(MujocoCamera):
    def __init__(
        self,
        camera_name,
        height=80,
        width=80,
        corruptor=None,
        depth=False,
        preprocess=False,
        grayscale=False,
        segmentation=False,
        scene_option=None,
    ):
        """
        Initialize a : class : ` MujocoCamera `.

        Args:
            camera_name: Name of the camera to use
            height: Height of the camera in pixels
            width: Width of the camera in pixels ( default 80 )
            corruptor: Corruptor to use for the camera ( default None )
            depth: True if the camera should be depth - corrected ( default False )
            preprocess: True if the camera should be pre - processed ( default False )
            grayscale: True if the camera should return a grayscale image ( default False )
            segmentation: True if the camera should return a segmented image ( default False )
            scene_option: set options for the MuJoCo scene.
        """
        super().__init__(camera_name, height, width)
        self._dtype = np.uint8
        self._n_channels = 1 if segmentation else 3
        self._preprocess = preprocess
        self.scene_option = scene_option
        self.segmentation = segmentation

    def _callable(self, physics):
        """
        Returns a callable that renders the image. This is used to implement : py : meth : ` render `

        Args:
            physics: The : py : class : ` Physics ` to render.

        Returns:
            A callable that renders the image and returns it as a 3D array of shape ( height width depth
        """

        def get_image():
            image = physics.render(
                self._height,
                self._width,
                self._camera_name,
                depth=self._depth,
                scene_option=self.scene_option,
                segmentation=self.segmentation,
            )
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
    @composer.observable
    def joint_positions(self):
        """
        Returns a observable sequence of joint positions.


        Returns:
            observable sequence of joint positions
        """
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qpos", all_joints)

    @composer.observable
    def joint_velocities(self):
        """
        Returns a observable sequence of joint positions..


        Returns:
            Observable sequence of joint velocities
        """
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qvel", all_joints)
