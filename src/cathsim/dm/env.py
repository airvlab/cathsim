import math
import numpy as np
import trimesh
import random
from typing import Union, Optional


from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco import wrapper, engine
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable

from cathsim.dm.components import Phantom
from cathsim.dm.components import Guidewire, Tip
from cathsim.dm.utils import distance
from cathsim.dm.observables import CameraObservable
from cathsim.dm.utils import filter_mask, get_env_config
from cathsim.dm.visualization import (
    point2pixel,
    create_camera_matrix,
)


env_config = get_env_config()

OPTION = env_config["option"]
OPTION_FLAG = OPTION.pop("flag")
COMPILER = env_config["compiler"]
GUIDEWIRE_CONFIG = env_config["guidewire"]

DEFAULT_SITE = env_config["default"]["site"]
SKYBOX_TEXTURE = env_config["skybox_texture"]


random_state = np.random.RandomState(42)


def make_scene(geom_groups: list):
    """Make a scene option for the phantom. This is used to set the visibility of the different parts of the environment.

    1. Phantom
    2. Guidewire
    3. Tip

    Args:
        geom_groups: A list of geom groups to set visible

    Returns:
        A scene option

    Examples:
        >>> make_scene([1, 2])
        >>> make_scene([0, 1, 2])
    """
    scene_option = wrapper.MjvOption()
    scene_option.geomgroup = np.zeros_like(scene_option.geomgroup)
    for geom_group in geom_groups:
        scene_option.geomgroup[geom_group] = True
    return scene_option


def sample_points(
    mesh: trimesh.Trimesh, y_bounds: tuple, n_points: int = 10
) -> np.ndarray:
    """Sample points from a mesh.

    Args:
        mesh (trimesh.Trimesh): A trimesh mesh
        y_bounds (tuple): The bounds of the y axis
        n_points (int): The number of points to sample ( default : 10 )

    Returns:
        np.ndarray: A point sampled from the mesh
    """

    def is_within_limits(point):
        return y_bounds[0] < point[1] < y_bounds[1]

    while True:
        points = trimesh.sample.volume_mesh(mesh, n_points)
        valid_points = [point for point in points if is_within_limits(point)]

        if valid_points:
            return random.choice(valid_points)


callback_called = 0


class Scene(composer.Arena):
    def _build(self, name: str = "arena"):
        """Build the scene.

        This method is called by the composer.Arena constructor. It is responsible for
        constructing the MJCF model of the scene and adding it to the arena. It sets
        the main attributes of the MJCF root element, such as compiler options and
        default lighting.

        Args:
            name (str): Name of the arena ( default : "arena" )
        """
        super()._build(name=name)

        self._set_mjcf_attributes()
        self._add_cameras()
        self._add_assets_and_lights()

    def _set_mjcf_attributes(self):
        """Set the attributes for the mjcf root."""
        self._mjcf_root.compiler.set_attributes(**COMPILER)
        self._mjcf_root.option.set_attributes(**OPTION)
        self._mjcf_root.option.flag.set_attributes(**OPTION_FLAG)
        self._mjcf_root.default.site.set_attributes(**DEFAULT_SITE)

    def _add_cameras(self):
        """Add cameras to the scene."""
        self.top_camera = self.add_camera(
            "top_camera",
            [-0.03, 0.125, 0.25],
            quat=[1, 0, 0, 0],
        )
        self.top_camera_close = self.add_camera(
            "top_camera_close",
            [-0.03, 0.125, 0.065],
            quat=[1, 0, 0, 0],
        )
        self.side_camera = self.add_camera(
            "side", [-0.22, 0.105, 0.03], quat=[0.5, 0.5, -0.5, -0.5]
        )

    def _add_assets_and_lights(self):
        """Add assets and lights to the scene."""
        self._mjcf_root.asset.add("texture", **SKYBOX_TEXTURE)
        self.add_light(pos=[0, 0, 10], dir=[20, 20, -20])

    def add_light(
        self, pos: list = [0, 0, 0], dir: list = [0, 0, 0], **kwargs
    ) -> mjcf.Element:
        """Add a light object to the scene.

        For more information about the light object, see the mujoco documentation:
        https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-light

        Args:
            pos (list): Position of the light ( default : [0, 0, 0] )
            dir (list): Direction of the light ( default : [0, 0, 0] )
            **kwargs: Additional arguments for the light

        Returns:
            mjcf.Element: The light element
        """
        light = self._mjcf_root.worldbody.add(
            "light", pos=pos, dir=dir, castshadow=False, **kwargs
        )

        return light

    def add_camera(self, name: str, pos: list = [0, 0, 0], **kwargs) -> mjcf.Element:
        """Add a camera object to the scene.

        For more information about the camera object, see the mujoco documentation:
        https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera

        Args:
            name (str): Name of the camera
            pos (list): Position of the camera ( default : [0, 0, 0] )
            **kwargs: Additional arguments for the camera

        Returns:
            mjcf.Element: The camera element
        """
        camera = self._mjcf_root.worldbody.add("camera", name=name, pos=pos, **kwargs)
        return camera

    def add_site(self, name: str, pos: list = [0, 0, 0], **kwargs) -> mjcf.Element:
        """Add a site object to the scene.

        For more information about the site object, see the mujoco documentation:
        https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site

        Args:
            name (str): Name of the site
            pos (list): Position of the site ( default : [0, 0, 0] )

        Returns:
            mjcf.Element: The site element
        """
        site = self._mjcf_root.worldbody.add("site", name=name, pos=pos, **kwargs)
        return site

    @property
    def cameras(self):
        """The get_cameras property."""
        return self._mjcf_root.find_all("camera")


class UniformSphere(variation.Variation):
    """Uniformly samples points from a sphere.

    This class samples points uniformly from a sphere using spherical coordinates
    and then converts them to Cartesian coordinates.

    The conversion from spherical to Cartesian coordinates is done using the following formulas:

    .. math::
       x = r \times \sin(\phi) \times \cos(\theta)
       y = r \times \sin(\phi) \times \sin(\theta)
       z = r \times \cos(\phi)

    Where:
        - r is the radius of the sphere.
        - theta (θ) is the azimuthal angle, ranging from 0 to 2π.
        - phi (φ) is the polar angle, ranging from 0 to π.

    Note:
        The cube root transformation for the radius ensures that points are uniformly
        distributed within the sphere, not just on its surface.
    """

    def __init__(self, radius: float = 0.001):
        """Uniformly sample a point on a sphere.

        Args:
            radius (float): Radius of the sphere ( default : 0.001 )
        """
        self._radius = radius

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        theta = 2 * math.pi * random.random()
        phi = math.acos(2 * random.random() - 1)
        r = self._radius * (random.random() ** (1 / 3))

        # Convert spherical to cartesian
        x_pos = r * math.sin(phi) * math.cos(theta)
        y_pos = r * math.sin(phi) * math.sin(theta)
        z_pos = r * math.cos(phi)

        return (x_pos, y_pos, z_pos)


class Navigate(composer.Task):
    """The task for the navigation environment.

    Args:
        phantom: The phantom entity to use
        guidewire: the guidewire entity to use
        tip: The tip entity to use for the tip ( default : None )
        delta: Minimum distance threshold for the success reward ( default : 0.004 )
        dense_reward: If True, the reward is the distance to the target ( default : True )
        success_reward: Success reward ( default : 10.0 )
        use_pixels: Add pixels to the observation ( default : False )
        use_segment: Add guidewire segmentation to the observation ( default : False )
        use_phantom_segment: Add phantom segmentation to the observation ( default : False )
        image_size: The size of the image ( default : 80 )
        sample_target: Weather or not to sample the target ( default : False )
        visualize_sites: If True, the sites will be rendered ( default : False )
        target_from_sites: If True, the target will be sampled from sites ( default : True )
        random_init_distance: The distance from the center to sample the initial pose ( default : 0.001 )
        target: The target to use. Can be a string or a numpy array ( default : None )
    """

    def __init__(
        self,
        phantom: composer.Entity = None,
        guidewire: composer.Entity = None,
        tip: composer.Entity = None,
        delta: float = 0.004,
        dense_reward: bool = True,
        success_reward: float = 10.0,
        use_pixels: bool = False,
        use_side: bool = False,
        use_segment: bool = False,
        use_phantom_segment: bool = False,
        image_size: int = 80,
        sample_target: bool = False,
        visualize_sites: bool = False,
        visualize_target: bool = False,
        target_from_sites: bool = True,
        random_init_distance: float = 0.001,
        target: Union[str, np.ndarray] = None,
    ):
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_pixels = use_pixels
        self.use_side = use_side
        self.use_segment = use_segment
        self.use_phantom_segment = use_phantom_segment
        self.image_size = image_size
        self.visualize_sites = visualize_sites
        self.visualize_target = visualize_target
        self.sample_target = sample_target
        self.target_from_sites = target_from_sites
        self.sampling_bounds = (0.0954, 0.1342)
        self.random_init_distance = random_init_distance

        self._setup_arena_and_attachments(phantom, guidewire, tip)

        self._configure_poses_and_variators()

        self._setup_observables()

        self._setup_visualizations()

        self.control_timestep = env_config["num_substeps"] * self.physics_timestep
        self.success = False
        self.camera_matrix = self.get_camera_matrix(
            image_size=self.image_size, camera_name="top_camera"
        )
        self.set_target(target)

    def _setup_arena_and_attachments(
        self, phantom: composer.Entity, guidewire: composer.Entity, tip: composer.Entity
    ):
        """Setup the arena and attachments.

        This method is responsible for setting up the arena and attaching the entities.

        Args:
            phantom (composer.Entity): The phantom entity to use
            guidewire (composer.Entity): The guidewire entity to use
            tip (composer.Entity): The tip entity to use for the tip
        """
        self._arena = Scene("arena")
        self._phantom = phantom or composer.Entity()
        self._arena.attach(self._phantom)

        if guidewire is not None:
            self._guidewire = guidewire
            self._arena.attach(self._guidewire)
            if tip is not None:
                self._tip = tip
                self._guidewire.attach(self._tip)

    def _configure_poses_and_variators(self):
        """Setup the initial poses and variators."""
        self._guidewire_initial_pose = UniformSphere(radius=self.random_init_distance)
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

    def _setup_observables(self):
        """Setup task observables."""
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0001))
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.0001))

        self._task_observables = {}

        if self.use_pixels:
            self._task_observables["pixels"] = CameraObservable(
                camera_name="top_camera",
                width=self.image_size,
                height=self.image_size,
            )
            if self.use_side:
                guidewire_option = make_scene([1, 2])
                self._task_observables["side"] = CameraObservable(
                    camera_name="side",
                    width=self.image_size,
                    height=self.image_size,
                    scene_option=guidewire_option,
                    segmentation=True,
                )

        if self.use_segment:
            guidewire_option = make_scene([1, 2])

            self._task_observables["guidewire"] = CameraObservable(
                camera_name="top_camera",
                height=self.image_size,
                width=self.image_size,
                scene_option=guidewire_option,
                segmentation=True,
            )

        if self.use_phantom_segment:
            phantom_option = make_scene([0])
            self._task_observables["phantom"] = CameraObservable(
                camera_name="top_camera",
                height=self.image_size,
                width=self.image_size,
                scene_option=phantom_option,
                segmentation=True,
            )

        self._task_observables["joint_pos"] = observable.Generic(
            self.get_joint_positions
        )
        self._task_observables["joint_vel"] = observable.Generic(
            self.get_joint_velocities
        )
        self._task_observables["joint_pos"].corruptor = pos_corrptor

        self._task_observables["joint_vel"].corruptor = vel_corruptor

        for obs in self._task_observables.values():
            obs.enabled = True

    def _setup_visualizations(self):
        """Setup visual elements for the task."""
        if self.visualize_sites:
            sites = self._phantom._mjcf_root.find_all("site")
            for site in sites:
                site.rgba = [1, 0, 0, 1]

    @property
    def root_entity(self):
        """The root_entity property."""
        return self._arena

    @property
    def task_observables(self):
        """The task_observables property."""
        return self._task_observables

    def set_target(self, target) -> None:
        """Set the target position."""
        if isinstance(target, str):
            sites = self._phantom.sites
            assert (
                target in sites
            ), f"Target site not found. Valid sites are: {sites.keys()}"
            target = sites[target]

        if self.visualize_target:
            if not hasattr(self, "target_site"):
                self.target_site = self._arena.add_site(
                    "target",
                    pos=target,
                    size=[0.003],
                    rgba=[1, 0, 0, 1],
                )
            else:
                self.target_site.pos = target

        self.target_pos = target

    def initialize_episode_mjcf(self, random_state):
        """Initialize the episode mjcf."""
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        """Initialize the episode."""

        self._physics_variator.apply_variations(physics, random_state)

        # Set initial guidewire pose
        guidewire_pose = variation.evaluate(
            self._guidewire_initial_pose, random_state=random_state
        )
        self._guidewire.set_pose(physics, position=guidewire_pose)

        self.success = False
        if self.sample_target:
            self.set_target(self.get_random_target(physics))

    def get_reward(self, physics):
        """Get the reward from the environment."""
        self.head_pos = self.get_head_pos(physics)

        d = distance(self.head_pos, self.target_pos)
        is_successful = d < self.delta

        # Calculate reward based on success and reward type
        if self.dense_reward:
            reward = self.success_reward if is_successful else -d
        else:
            reward = self.success_reward if is_successful else -1.0

        self.success = is_successful
        return reward

    def should_terminate_episode(self, physics):
        """Check if the episode should terminate."""
        return self.success

    def get_head_pos(self, physics):
        """Get the position of the head of the guidewire."""
        return physics.named.data.geom_xpos[-1]

    def get_target_pos(self, physics):
        """Get the position of the target."""
        return self.target_pos

    def get_joint_positions(self, physics):
        """Get the joint positions."""
        positions = physics.named.data.qpos
        return positions

    def get_joint_velocities(self, physics):
        """Get the joint velocities."""
        velocities = physics.named.data.qvel
        return velocities

    def get_total_force(self, physics):
        """Get the force magnitude."""
        forces = physics.data.qfrc_constraint[0:3]
        forces = np.linalg.norm(forces)
        return forces

    def get_contact_forces(
        self,
        physics: engine.Physics,
        threshold: float = 0.001,
        to_pixels: bool = True,
        image_size: int = 80,
    ) -> dict:
        """Get the contact forces for each contact.

        Get the contact forces for each contact. The contact forces are filtered
        by a threshold and converted to pixels if needed.

        Args:
            physics (engine.Physics): A dm_control physics object
            threshold (float): The threshold to filter the forces ( default : 0.01 )
            to_pixels (bool): Convert the forces to pixels ( default : True )
            image_size (int): The size of the image ( default : 80 )

        Returns:
            dict: A dictionary containing the positions and forces
        """
        data = physics.data
        forces = {"pos": [], "force": []}

        for i in range(data.ncon):
            if data.contact[i].dist < 0.002:
                force = data.contact_force(i)[0][0]
                if abs(force) > threshold:
                    forces["force"].append(force)
                    pos = data.contact[i].pos
                    if to_pixels:
                        pos = point2pixel(pos, self.camera_matrix)
                    forces["pos"].append(pos)

        return forces

    def get_camera_matrix(
        self,
        image_size: Optional[int] = None,
        camera_name: str = "top_camera",
    ) -> np.ndarray:
        cameras = self._arena.mjcf_model.find_all("camera")
        camera = next((cam for cam in cameras if cam.name == camera_name), None)

        if camera is None:
            raise ValueError(f"No camera found with the name: {camera_name}")

        assert camera.quat is not None, "Camera quaternion is None"

        image_size = image_size or self.image_size
        camera_matrix = create_camera_matrix(
            image_size=image_size,
            pos=camera.pos,
            quat=camera.quat,
        )

        return camera_matrix

    def get_phantom_mask(self, physics, image_size: int = None, camera_id=0):
        """Get the phantom mask."""
        scene_option = make_scene([0])
        if image_size is None:
            image_size = self.image_size
        image = physics.render(
            height=image_size,
            width=image_size,
            camera_id=camera_id,
            scene_option=scene_option,
        )
        mask = filter_mask(image)
        return mask

    def get_guidewire_mask(self, physics, image_size: int = None, camera_id=0):
        """Get the guidewire mask."""
        scene_option = make_scene([1, 2])
        if image_size is None:
            image_size = self.image_size
        image = physics.render(
            height=image_size,
            width=image_size,
            camera_id=camera_id,
            scene_option=scene_option,
        )
        mask = filter_mask(image)
        return mask

    def get_random_target(self, physics):
        """Get a random target based on conditions."""

        # If targets are fetched from predefined sites
        if self.target_from_sites:
            sites = self._phantom.sites
            site = np.random.choice(list(sites.keys()))
            return sites[site]

        # Else, get targets from a mesh
        mesh = trimesh.load_mesh(self._phantom.simplified, scale=0.9)
        return sample_points(mesh, self.sampling_bounds)

    def get_guidewire_geom_pos(self, physics: engine.Physics) -> list[np.ndarray]:
        """Get the guidewire geometry positions.

        Args:
            physics (engine.Physics): DM control physics object

        Returns:
            list[np.ndarray]: A list of positions of the guidewire geometries
        """

        model = physics.copy().model

        # Collect IDs of all guidewire geometries
        guidewire_geom_ids = []
        for i in range(model.ngeom):
            geom_name = model.geom(i).name
            if "guidewire" in geom_name:
                guidewire_geom_ids.append(model.geom(i).id)

        # Get positions based on IDs
        guidewire_geom_pos = [physics.data.geom_xpos[i] for i in guidewire_geom_ids]

        return guidewire_geom_pos

    def get_jacobian_for_geom_id(self, physics: engine.Physics, geom_id: int):
        jac_pos = np.zeros((3, physics.model.nv))
        jac_rot = np.zeros((3, physics.model.nv))

        # get Jacobian of specific geometry
        mjlib.mj_jacGeom(
            physics.model.ptr,
            physics.data.ptr,
            jac_pos,
            jac_rot,
            geom_id,
        )
        return jac_pos, jac_rot

    def get_guidewire_geom_info(self, physics: engine.Physics) -> list[np.ndarray]:
        if hasattr(self, "guidewire_bodies"):
            return self.guidewire_bodies

        model = physics.copy().model

        # Collect IDs of all guidewire geometries
        guidewire_geom_ids = []
        for i in range(model.nbody):
            geom_name = model.body(i).name
            contains_guidewire = "guidewire" in geom_name
            is_part = "body" in geom_name
            if contains_guidewire and is_part:
                guidewire_geom_ids.append(model.geom(i).id)

        # Get positions and Jacobians based on IDs
        guidewire_geom_info = []
        for i in guidewire_geom_ids:
            position = physics.data.geom_xpos[i]
            guidewire_geom_info.append((i, position))

        self.guidewire_bodies = guidewire_geom_info
        return self.guidewire_bodies

    def before_step(self, physics, action, random_state):
        guidewire_geom_info = self.get_guidewire_geom_info(physics)

        for i, pos in guidewire_geom_info:
            f = np.random.uniform(-0.1, 0.1, size=(3, 1))
            torque = np.zeros_like(f)
            mujoco.mj_applyFT(
                physics.model.ptr,
                physics.data.ptr,
                f,
                torque,
                pos,
                i,
                physics.data.qfrc_applied,
            )

        del random_state
        physics.set_control(action)

    @staticmethod
    def get_jacobian(physics: engine.Physics):
        jac_pos = np.zeros((3, physics.model.nv))
        jac_rot = np.zeros((3, physics.model.nv))

        mjlib.mj_jacGeom(
            physics.model.ptr,
            physics.data.ptr,
            jac_pos,
            jac_rot,
            physics.model.name2id("guidewire/tip/head", "geom"),
        )

        print("Jac Pos: ", jac_pos.shape)
        print("Jac Rot: ", jac_rot.shape)
        return jac_pos


def cartesian_to_joint_force(force: np.ndarray):
    pass


def make_dm_env(
    phantom: str = "phantom3",
    target: str = "bca",
    **kwargs,
) -> composer.Environment:
    """Makes a dm_control environment given a configuration.

    Args:
      phantom: str:  (Default value = "phantom3") The phantom to use
      **kwargs: Additional arguments for the environment

    Returns:
        composer.Environment: The environment

    """

    phantom = Phantom(phantom + ".xml")
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        target=target,
        **kwargs,
    )
    env = composer.Environment(
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )

    return env


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2

    data_path = Path.cwd() / "data"

    def save_guidewire_reconstruction(
        t: int,
        top: np.ndarray,
        side: np.ndarray,
        geom_pos: np.ndarray,
        path: Path = data_path / "guidewire-reconstruction-2",
    ):
        if not path.exists():
            path.mkdir(parents=True)

        plt.imsave(path / f"{t}_top.png", top[:, :, 0], cmap="gray")
        plt.imsave(path / f"{t}_side.png", side[:, :, 0], cmap="gray")
        np.save(path / f"{t}_geom_pos", geom_pos)

    env = make_dm_env(
        phantom="phantom3",
        use_pixels=True,
        use_segment=True,
        use_side=True,
        target="bca",
        image_size=480,
        visualize_sites=False,
        visualize_target=True,
        sample_target=True,
        target_from_sites=False,
    )

    env._task.get_guidewire_geom_pos(env.physics)
    physics = env.physics
    print("qpos", physics.data.qpos.shape)
    image_size = 480

    def random_policy(time_step):
        del time_step
        return [0.4, random.random() * 2 - 1]

    for episode in range(1):
        time_step = env.reset()
        # print(env._task.target_pos)
        # print(env._task.get_head_pos(env._physics))
        for step in range(200):
            action = random_policy(time_step)
            if step > 30:
                action = np.zeros_like(action)
            top = time_step.observation["guidewire"]
            cv2.imshow("top", top)
            cv2.waitKey(1)
            time_step = env.step(action)
