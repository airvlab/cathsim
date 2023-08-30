import math
from pathlib import Path

from dm_control import mjcf
from dm_control import composer

from cathsim.dm.observables import JointObservables
from cathsim.dm.utils import get_env_config


guidewire_config = get_env_config("guidewire")
guidewire_default = guidewire_config["default"]

tip_config = get_env_config("tip")
tip_default = tip_config["default"]

SCALE = guidewire_config["scale"]
RGBA = guidewire_config["rgba"]
BODY_DIAMETER = guidewire_config["diameter"]


def get_body_properties(
    scale: float, body_diameter: float, sphere_to_cylinder: float = 1.5
):
    sphere_radius = (body_diameter / 2) * scale
    cylinder_height = sphere_radius * sphere_to_cylinder
    offset = sphere_radius + cylinder_height * 2
    return sphere_radius, cylinder_height, offset


SPHERE_RADIUS, CYLINDER_HEIGHT, OFFSET = get_body_properties(
    scale=guidewire_config["scale"],
    body_diameter=guidewire_config["diameter"],
)


class BaseBody(composer.Entity):
    def add_body(
        self,
        n: int = 0,
        parent: mjcf.Element = None,
        stiffness: float = None,
        name: str = None,
    ):
        """Add a body to the guidewire.

        Args:
            n (int): The index of the body to add
            parent (mjcf.Element): The parent element to add the body to
            stiffness (float): Stiffness of the joint
            name (str): Name of the body/joint/geom
        """
        child = parent.add("body", name=f"{name}_body_{n}", pos=[0, 0, OFFSET])
        child.add("geom", name=f"geom_{n}")
        j0 = child.add("joint", name=f"{name}_J0_{n}", axis=[1, 0, 0])
        j1 = child.add("joint", name=f"{name}_J1_{n}", axis=[0, 1, 0])
        if stiffness is not None:
            j0.stiffness = stiffness
            j1.stiffness = stiffness

        return child

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all("joint"))


class Guidewire(BaseBody):
    """Guidewire class"""

    def _build(self, n_bodies: int = 80):
        """Build the guidewire.

        Set the default values, add bodies and joints, and add actuators.

        Args:
            n_bodies (int): Number of bodies to add to the guidewire
        """
        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies
        self._n_bodies = n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._set_defaults()
        self._set_bodies_and_joints()
        self._set_actuators()

    def _set_defaults(self):
        """Set the default values for the guidewire."""
        self._mjcf_root.default.geom.set_attributes(
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            **guidewire_default["geom"],
        )

        self._mjcf_root.default.joint.set_attributes(
            pos=[0, 0, -OFFSET / 2],
            **guidewire_default["joint"],
        )

        self._mjcf_root.default.site.set_attributes(
            size=[SPHERE_RADIUS],
            **guidewire_default["site"],
        )

        self._mjcf_root.default.velocity.set_attributes(
            **guidewire_default["velocity"],
        )

    def _set_bodies_and_joints(self):
        """Set the bodies and joints of the guidewire."""
        parent = self._mjcf_root.worldbody.add(
            "body",
            name="guidewire_body_0",
            euler=[-math.pi / 2, 0, math.pi],
            pos=[0, -(self._length - 0.015), 0],
        )
        parent.add(
            "geom",
            name="guidewire_geom_0",
        )
        parent.add(
            "joint",
            type="slide",
            name="slider",
            range=[-0, 0.2],
            stiffness=0,
            damping=2,
        )
        parent.add(
            "joint",
            type="hinge",
            name="rotator",
            stiffness=0,
            damping=2,
        )

        stiffness = self._mjcf_root.default.joint.stiffness
        for n in range(1, self._n_bodies):
            parent = self.add_body(n, parent, stiffness=stiffness, name="guidewire")
            stiffness *= 0.995
        self._tip_site = parent.add("site", name="tip_site", pos=[0, 0, OFFSET])

    def _set_actuators(self):
        """Set the actuators of the guidewire."""
        kp = 40
        self._mjcf_root.actuator.add(
            "velocity",
            joint="slider",
            name="slider_actuator",
        )
        self._mjcf_root.actuator.add(
            "general",
            joint="rotator",
            name="rotator_actuator",
            dyntype=None,
            gaintype="fixed",
            biastype="None",
            dynprm=[1, 0, 0],
            gainprm=[kp, 0, 0],
            biasprm=[0, kp, 0],
        )

    @property
    def attachment_site(self):
        """The attachment site of the guidewire. Useful for attaching the tip to the guidewire."""
        return self._tip_site

    def _build_observables(self):
        """Build the observables of the guidewire."""
        return JointObservables(self)

    @property
    def actuators(self):
        """Get the actuators of the guidewire."""
        return tuple(self._mjcf_root.find_all("actuator"))

    def save_model(self, path: Path):
        """Save the guidewire model to an `.xml` file.

        Usefull for debugging, exporting, and visualizing the guidewire.

        Args:
            path (Path): Path to save the model to
        """
        if path.suffix is None:
            path = path / "guidewire.xml"
        with open(path, "w") as file:
            file.write(self.mjcf_model.to_xml_string("guidewire"))


class Tip(BaseBody):
    def _build(self, name: str = "tip", n_bodies: int = 3):
        """Build the tip of the guidewire.

        Args:
            name (str): Name of the tip (Will be removed in the future)
            n_bodies (int): Number of bodies to add to the tip
        """
        self._mjcf_root = mjcf.RootElement(model=name)
        self._n_bodies = n_bodies

        self._setup_defaults()
        self._setup_bodies_and_joints()

        self.head_geom.name = "head"

    def _setup_defaults(self):
        """Set the default values for the tip."""
        self._mjcf_root.default.geom.set_attributes(
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            **tip_default["geom"],
        )

        self._mjcf_root.default.joint.set_attributes(
            pos=[0, 0, -OFFSET / 2],
            springref=math.pi / 3 / self._n_bodies,
            **tip_default["joint"],
        )

    def _setup_bodies_and_joints(self):
        """Setup the bodies and joints of the tip."""
        parent = self._mjcf_root.worldbody.add(
            "body",
            name="tip_body_0",
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add("geom", name="tip_geom_0")
        parent.add("joint", name="tip_J0_0", axis=[0, 0, 1])
        parent.add("joint", name="tip_J1_0", axis=[0, 1, 0])

        for n in range(1, self._n_bodies):
            parent = self.add_body(n, parent, name="tip")

    def _build_observables(self):
        """Setup the observables of the tip."""
        return JointObservables(self)

    @property
    def head_geom(self):
        """Get the head geom of the tip."""
        return self._mjcf_root.find_all("geom")[-1]


if __name__ == "__main__":
    guidewire = Guidewire()
    guidewire.mjcf_model.compiler.set_attributes(autolimits=True, angle="radian")
    mjcf.Physics.from_mjcf_model(guidewire.mjcf_model)
    guidewire.save_model(Path.cwd() / "guidewire.xml")
