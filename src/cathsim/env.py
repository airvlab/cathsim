import numpy as np
import trimesh
import random
from typing import Union


from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable

from cathsim.phantom import Phantom
from cathsim.utils import distance
from cathsim.guidewire import Guidewire, Tip
from cathsim.observables import CameraObservable

from cathsim.utils import filter_mask, point2pixel, get_env_config

env_config = get_env_config()

option = env_config["option"]
option_flag = option.pop("flag")
compiler = env_config["compiler"]
visual = env_config["visual"]
visual_global = visual.pop("global")
guidewire_config = env_config["guidewire"]

BODY_DIAMETER = guidewire_config["diameter"] * guidewire_config["scale"]
SPHERE_RADIUS = (BODY_DIAMETER / 2) * guidewire_config["scale"]
CYLINDER_HEIGHT = SPHERE_RADIUS * guidewire_config["sphere_to_cylinder_ratio"]
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2


random_state = np.random.RandomState(42)


def make_scene(geom_groups: list):
    """Make a scene option for the phantom. This is used to set the visibility of the different parts of the environment.

    Args:
        geom_groups: A list of geom groups to set visible
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

    This function samples points from a mesh and returns a point that is within the y_bounds.

    Args:
        mesh: A trimesh mesh
        y_bounds: The bounds of the y axis
        n_points: The number of points to sample ( default : 10 )

    Returns:
        np.ndarray: A point that is within the y_bounds
    """

    def is_within_limits(point: list) -> bool:
        """Check if a point is within the y_bounds."""
        return y_bounds[0] < point[1] < y_bounds[1]

    while True:
        points = trimesh.sample.volume_mesh(mesh, n_points)
        if len(points) == 0:
            continue
        valid_points = [point for point in points if is_within_limits(point)]
        if len(valid_points) == 0:
            continue
        elif len(valid_points) == 1:
            return valid_points[0]
        else:
            return random.choice(valid_points)


class Scene(composer.Arena):
    """Sets up the scene for the environment.

    Args:
        name: Name of the scene ( default : "arena" )
        render_site: If True, the sites will be rendered ( default : False )
    """

    def _build(
        self,
        name: str = "arena",
        render_site: bool = False,
    ):
        super()._build(name=name)

        self._mjcf_root.compiler.set_attributes(**compiler)
        self._mjcf_root.option.set_attributes(**option)
        self._mjcf_root.option.flag.set_attributes(**option_flag)
        self._mjcf_root.visual.set_attributes(**visual)

        self._top_camera = self.add_camera(
            "top_camera", [-0.03, 0.125, 0.15], [0, 0, 0]
        )
        self._top_camera_close = self.add_camera(
            "top_camera_close", [-0.03, 0.125, 0.065], [0, 0, 0]
        )
        self._mjcf_root.default.site.set_attributes(
            type="sphere",
            size=[0.002],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self._mjcf_root.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
            width=256,
            height=256,
        )

        self.add_light(pos=[0, 0, 10], dir=[20, 20, -20], castshadow=False)

    def add_light(
        self, pos: list = [0, 0, 0], dir: list = [0, 0, 0], castshadow: bool = False
    ) -> mjcf.Element:
        """Add a light object to the scene.

        Args:
            pos: Position of the light ( default : [0, 0, 0] )
            dir: Direction of the light ( default : [0, 0, 0] )
            castshadow: If True, the light will cast shadows ( default : False )

        Returns:
            mjcf.Element: The light element
        """
        light = self._mjcf_root.worldbody.add(
            "light", pos=pos, dir=dir, castshadow=castshadow
        )
        return light

    def add_camera(
        self, name: str, pos: list = [0, 0, 0], euler: list = [0, 0, 0]
    ) -> mjcf.Element:
        """Add a camera object to the scene.

        Args:
            name: Name of the camera
            pos: Position of the camera ( default : [0, 0, 0] )
            euler: Euler angles of the camera ( default : [0, 0, 0] )

        Returns:
            mjcf.Element: The camera element
        """
        camera = self._mjcf_root.worldbody.add(
            "camera", name=name, pos=pos, euler=euler
        )
        return camera

    def add_site(self, name: str, pos: list = [0, 0, 0]) -> mjcf.Element:
        """Add a site object to the scene. This is used to visualize/set the target.

        Args:
            name: Name of the site
            pos: position of the site ( default : [0, 0, 0] )

        Returns:
            mjcf.Element: The site element
        """
        site = self._mjcf_root.worldbody.add("site", name=name, pos=pos)
        return site


class UniformCircle(variation.Variation):
    """A uniform circle variation. Allows to sample a point from a uniform circle.

    Args:
        x_range: The range of the x axis ( default : (-0.001, 0.001) )
        y_range: The range of the y axis ( default : (-0.001, 0.001) )
        z_range: The range of the z axis ( default : (-0.001, 0.001) )
    """

    def __init__(
        self,
        x_range: tuple[int] = (-0.001, 0.001),
        y_range: tuple[int] = (-0.001, 0.001),
        z_range: tuple[int] = (-0.001, 0.001),
    ):
        self._x_distrib = distributions.Uniform(*x_range)
        self._y_distrib = distributions.Uniform(*y_range)
        self._z_distrib = distributions.Uniform(*z_range)

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        x_pos = variation.evaluate(self._x_distrib, random_state=random_state)
        y_pos = variation.evaluate(self._y_distrib, random_state=random_state)
        z_pos = variation.evaluate(self._z_distrib, random_state=random_state)
        return (x_pos, y_pos, z_pos)


class Navigate(composer.Task):
    """The task for the navigation environment.

    Args:
        phantom: The phantom entity to use
        guidewire: The guidewire entity to use
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
        use_segment: bool = False,
        use_phantom_segment: bool = False,
        image_size: int = 80,
        sample_target: bool = False,
        visualize_sites: bool = False,
        target_from_sites: bool = True,
        random_init_distance: float = 0.001,
        target: Union[str, np.ndarray] = None,
    ):
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_pixels = use_pixels
        self.use_segment = use_segment
        self.use_phantom_segment = use_phantom_segment
        self.image_size = image_size
        self.sample_target = sample_target
        self.visualize_sites = visualize_sites
        self.target_from_sites = target_from_sites
        self.random_init_distance = random_init_distance

        self._arena = Scene("arena")
        if phantom is not None:
            self._phantom = phantom
            self._arena.attach(self._phantom)
        if guidewire is not None:
            self._guidewire = guidewire
            if tip is not None:
                self._tip = tip
                self._guidewire.attach(self._tip)
            self._arena.attach(self._guidewire)

        # Configure initial poses
        self._guidewire_initial_pose = UniformCircle(
            x_range=(-random_init_distance, random_init_distance),
            y_range=(-random_init_distance, random_init_distance),
            z_range=(-random_init_distance, random_init_distance),
        )

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0001))
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.0001))

        self._task_observables = {}

        if self.use_pixels:
            self._task_observables["pixels"] = CameraObservable(
                camera_name="top_camera",
                width=image_size,
                height=image_size,
            )

        if self.use_segment:
            guidewire_option = make_scene([1, 2])

            self._task_observables["guidewire"] = CameraObservable(
                camera_name="top_camera",
                height=image_size,
                width=image_size,
                scene_option=guidewire_option,
                segmentation=True,
            )

        if self.use_phantom_segment:
            phantom_option = make_scene([0])
            self._task_observables["phantom"] = CameraObservable(
                camera_name="top_camera",
                height=image_size,
                width=image_size,
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

        self.control_timestep = env_config["num_substeps"] * self.physics_timestep

        self.success = False

        self.set_target(target)
        self.camera_matrix = None

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

    @property
    def target_pos(self):
        """The target_pos property."""
        return self._target_pos

    def set_target(self, target) -> None:
        """Set the target position."""
        if type(target) is str:
            sites = self._phantom.sites
            assert (
                target in sites
            ), f"Target site not found. Valid sites are: {sites.keys()}"
            target = sites[target]
        self._target_pos = target

    def initialize_episode_mjcf(self, random_state):
        """Initialize the episode mjcf."""
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        """Initialize the episode."""
        if self.camera_matrix is None:
            self.camera_matrix = self.get_camera_matrix(physics)
        self._physics_variator.apply_variations(physics, random_state)
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
        reward = self.compute_reward(self.head_pos, self._target_pos)
        return reward

    def should_terminate_episode(self, physics):
        """Check if the episode should terminate."""
        return self.success

    def get_head_pos(self, physics):
        """Get the position of the head of the guidewire."""
        return physics.named.data.geom_xpos[-1]

    def compute_reward(self, achieved_goal, desired_goal):
        """Compute the reward."""
        d = distance(achieved_goal, desired_goal)
        success = np.array(d < self.delta, dtype=bool)

        if self.dense_reward:
            reward = np.where(success, self.success_reward, -d)
        else:
            reward = np.where(success, self.success_reward, -1.0)
        self.success = success
        return reward

    def get_joint_positions(self, physics):
        """Get the joint positions."""
        positions = physics.named.data.qpos
        return positions

    def get_joint_velocities(self, physics):
        """Get the joint velocities."""
        velocities = physics.named.data.qvel
        return velocities

    def get_force(self, physics):
        """Get the force magnitude."""
        forces = physics.data.qfrc_constraint[0:3]
        forces = np.linalg.norm(forces)
        return forces

    def get_contact_forces(
        self,
        physics,
        threshold: float = 0.01,
        to_pixels: bool = True,
        image_size: int = 80,
    ):
        """Gets the contact forces for each contact point.

        Args:
            physics: The physics object
            threshold: The threshold for the force magnitude ( default : 0.01 )
            to_pixels: If True, the contact points will be converted to pixels ( default : True )
            image_size: The size of the image ( default : 80 )
        """
        if self.camera_matrix is None:
            self.camera_matrix = self.get_camera_matrix(physics, image_size)
        data = physics.data
        forces = {"pos": [], "force": []}
        for i in range(data.ncon):
            if data.contact[i].dist < 0.002:
                force = data.contact_force(i)[0][0]
                if abs(force) > threshold:
                    pass
                else:
                    forces["force"].append(force)
                    pos = data.contact[i].pos
                    if to_pixels is not None:
                        pos = point2pixel(pos, self.camera_matrix)
                    forces["pos"].append(pos)
        return forces

    def get_camera_matrix(self, physics, image_size: int = None, camera_id=0):
        """Get the camera matrix for the given camera."""
        from dm_control.mujoco.engine import Camera

        if image_size is None:
            image_size = self.image_size
        camera = Camera(
            physics, height=image_size, width=image_size, camera_id=camera_id
        )
        return camera.matrix

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
        """Get a random target."""
        if self.target_from_sites:
            sites = self._phantom.sites
            site = np.random.choice(list(sites.keys()))
            target = sites[site]
            return target
        mesh = trimesh.load_mesh(self._phantom.simplified, scale=0.9)
        return sample_points(mesh, (0.0954, 0.1342))

    def get_guidewire_geom_pos(self, physics):
        """Get the guidewire geom positions."""
        model = physics.copy().model
        guidewire_geom_ids = [
            model.geom(i).id
            for i in range(model.ngeom)
            if "guidewire" in model.geom(i).name
        ]
        guidewire_geom_pos = [physics.data.geom_xpos[i] for i in guidewire_geom_ids]
        return guidewire_geom_pos


def run_env(args=None):
    """Run the environment."""
    from argparse import ArgumentParser
    from cathsim.utils import launch

    parser = ArgumentParser()
    parser.add_argument("--interact", type=bool, default=True)
    parser.add_argument("--phantom", default="phantom3", type=str)
    parser.add_argument("--target", default="bca", type=str)

    parsed_args = parser.parse_args(args)

    phantom = Phantom(parsed_args.phantom + ".xml")

    tip = Tip()
    guidewire = Guidewire()

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target=parsed_args.target,
        visualize_sites=True,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    launch(env)


if __name__ == "__main__":
    phantom_name = "phantom3"
    phantom = Phantom(phantom_name + ".xml")
    tip = Tip()
    guidewire = Guidewire()

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target="bca",
        image_size=80,
        visualize_sites=False,
        sample_target=True,
        target_from_sites=False,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    env._task.get_guidewire_geom_pos(env.physics)
    exit()

    def random_policy(time_step):
        """

        :param time_step:

        """
        del time_step  # Unused
        return [0, 0]

    # loop 2 episodes of 2 steps
    for episode in range(2):
        time_step = env.reset()
        # print(env._task.target_pos)
        # print(env._task.get_head_pos(env._physics))
        print(env._task.get_camera_matrix(env.physics, 480))
        print(env._task.get_camera_matrix(env.physics, 80))
        exit()
        for step in range(2):
            action = random_policy(time_step)
            img = env.physics.render(height=480, width=480, camera_id=0)
            # plt.imsave("phantom_480.png", img)
            time_step = env.step(action)
