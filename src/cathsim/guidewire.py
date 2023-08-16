import math
from pathlib import Path

from dm_control import mjcf
from dm_control import composer
from cathsim.observables import JointObservables

from cathsim.utils import get_env_config


SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2

env_config = get_env_config()
guidewire_config = env_config["guidewire"]


class BaseBody(composer.Entity):
    def add_body(
        self,
        n: int = 0,
        parent: mjcf.Element = None,  # the parent body
        stiffness: float = None,  # the stiffness of the joint
        name: str = None,
    ):
        """
        Add a body to the MJCF element. This is a convenience method for procedurally adding guidewire bodies.

        Args:
            n: the index of the body to be added.
            parent: the parent body element. If None the body stiffness is set to the value of the parent bodystiffness.
            stiffness: body the stiffness of the joint.
            name: the name of the joint. Default is None.

        Returns:
            The newly added body element.
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
        """
        Build MJCF data and set attributes. This is called by __init__ to initialize the class.

        Args:
            n_bodies: Number of bodies to build ( default 80 )
        """

        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            group=1,
            rgba=guidewire_config["rgba"],
            type="capsule",
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            density=guidewire_config["density"],
            condim=guidewire_config["condim"],
            friction=guidewire_config["friction"],
            fluidshape="ellipsoid",
        )

        self._mjcf_root.default.joint.set_attributes(
            type="hinge",
            pos=[0, 0, -OFFSET / 2],
            ref=0,
            stiffness=guidewire_config["stiffness"],
            springref=0,
            armature=0.05,
            axis=[0, 0, 1],
        )

        self._mjcf_root.default.site.set_attributes(
            type="sphere",
            size=[SPHERE_RADIUS],
            rgba=[0.3, 0, 0, 0.0],
        )

        self._mjcf_root.default.velocity.set_attributes(
            ctrlrange=[-1, 1],
            forcerange=[-guidewire_config["force"], guidewire_config["force"]],
            kv=5,
        )

        parent = self._mjcf_root.worldbody.add(
            "body",
            name="guidewire_body_0",
            euler=[-math.pi / 2, 0, math.pi],
            pos=[0, -(self._length - 0.015), 0],
        )
        parent.add("geom", name="guidewire_geom_0")
        parent.add(
            "joint",
            type="slide",
            name="slider",
            range=[-0, 0.2],
            stiffness=0,
            damping=2,
        )
        parent.add("joint", type="hinge", name="rotator", stiffness=0, damping=2)
        self._mjcf_root.actuator.add("velocity", joint="slider", name="slider_actuator")
        kp = 40
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

        # make the main body
        stiffness = self._mjcf_root.default.joint.stiffness
        for n in range(1, n_bodies):
            parent = self.add_body(n, parent, stiffness=stiffness, name="guidewire")
            stiffness *= 0.995
        self._tip_site = parent.add("site", name="tip_site", pos=[0, 0, OFFSET])

    @property
    def attachment_site(self):
        return self._tip_site

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def _build_observables(self):
        return JointObservables(self)

    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all("actuator"))

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all("joint"))

    def save_model(self, path: Path):
        if path.suffix is None:
            path = path / "guidewire.xml"
        with open(path, "w") as file:
            file.write(self.mjcf_model.to_xml_string("guidewire"))


class Tip(BaseBody):
    def _build(self, name=None, n_bodies=3):
        """
        Build and return guidewire tip.

        Args:
            name: ( Default value = None )
            n_bodies: ( Default value = 3 )
        """
        if name is None:
            name = "tip"
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.default.geom.set_attributes(
            group=2,
            rgba=guidewire_config["rgba"],
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            type="capsule",
            condim=guidewire_config["condim"],
            friction=guidewire_config["friction"],
            fluidshape="ellipsoid",
        )

        self._mjcf_root.default.joint.set_attributes(
            type="hinge",
            pos=[0, 0, -OFFSET / 2],
            springref=math.pi / 3 / n_bodies,
            # ref=math.pi / 5 / n_bodies ,
            damping=0.5,
            stiffness=1,
            armature=0.05,
        )

        parent = self._mjcf_root.worldbody.add(
            "body",
            name="tip_body_0",
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add(
            "geom",
            name="tip_geom_0",
        )
        parent.add("joint", name="tip_J0_0", axis=[0, 0, 1])
        parent.add("joint", name="tip_J1_0", axis=[0, 1, 0])

        for n in range(1, n_bodies):
            parent = self.add_body(n, parent, name="tip")

        self.head_geom.name = "head"

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all("joint"))

    def _build_observables(self):
        return JointObservables(self)

    @property
    def head_geom(self):
        return self._mjcf_root.find_all("geom")[-1]


if __name__ == "__main__":
    guidewire = Guidewire()
    guidewire.mjcf_model.compiler.set_attributes(autolimits=True, angle="radian")
    mjcf.Physics.from_mjcf_model(guidewire.mjcf_model)
    guidewire.save_model(Path.cwd() / "gdw.xml")
