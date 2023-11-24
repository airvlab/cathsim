import math
from pathlib import Path

import dm_control
from dm_control import mjcf
from dm_control import composer

from cathsim.dm.components.base_models import BaseGuidewire
from cathsim.dm.utils import get_env_config

import xml.etree.ElementTree as ET
from xml.dom import minidom

import pprint

guidewire_config = get_env_config("guidewire_2")
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


class Guidewire(BaseGuidewire):
    def __str__(self):
        return pprint.pformat(self._mjcf_root)

    def _build(self):
        base_xml = f"""
        <mujoco model="cable">
        <extension>
                <plugin plugin="mujoco.elasticity.cable"/>
        </extension>
        <compiler meshdir="assets/meshes" angle="radian" autolimits="true"/>
        <option timestep="0.004" density="1055" viscosity="0.004" o_margin="0.004"
            integrator="implicitfast" cone="pyramidal" jacobian="sparse">
            <flag frictionloss="enable" multiccd="disable" gravity="enable"/>
        </option>
        <size memory="1G"/>
        <worldbody>
        <body name="core" pos="0 0 0" euler="0 0 {math.pi / 2}">
            <joint name="slider" type="slide" axis="1 0 0" stiffness="0" damping="0"/>
            <joint name="rotator" type="hinge" axis="1 0 0" stiffness="0" damping="5" armature="0.1"/>
            <composite prefix="test" type="cable" curve="s" count="31" size="0.3" offset="0.00025 0 0" initial="none">
                <geom type="capsule" size="0.0008" rgba="1 1 1 1" condim="1" density="6450" margin="0.02"/>
                <joint kind="main" damping="0.5" stiffness="0.01" margin="0.02"/>
                <plugin plugin="mujoco.elasticity.cable">
                    <config key="twist" value="1e6"/>
                    <config key="bend" value="1e9"/>
                    <config key="vmax" value="0.005"/>
                </plugin>
            </composite>
        </body>
        <body name="head" pos="0 0 0">
            <geom size=".01"/>
        </body>
        </worldbody>
        <equality>
            <connect name="right_boundary" body1="B_last" body2="head" anchor=".025 0 0"/>
        </equality>
        <contact>
            <exclude body1="B_last" body2="head"/>
        </contact>
        <actuator>
            <velocity name="slider_actuator" joint="slider"/>
            <general name="rotator_actuator" joint="rotator" biastype="none" dynprm="1 0 0" gainprm="40 0 0" biasprm="0 40 0"/>
        </actuator>
        </mujoco>
        """
        self._mjcf_root = mjcf.from_xml_string(base_xml)

        bodies = self._mjcf_root.find_all("body")
        for body in bodies:
            print(body)
        print(self._mjcf_root.to_xml_string())
        self._tip_site = self._mjcf_root.find("site", "tip_site")

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def head_geom(self):
        return self._mjcf_root.find_all("geom")[-1]

    # TODO: implement this
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

    # TODO: implement this
    def _set_actuators(self):
        pass

    # TODO: implement this
    def _set_bodies_and_joints(self):
        pass

    @property
    def length(self):
        """The length property."""

        return self._length


if __name__ == "__main__":
    from cathsim.dm import make_dm_env, Navigate
    from cathsim.dm.components import Phantom
    from dm_control import viewer
    guidewire = Guidewire()
    print(guidewire._mjcf_root.to_xml_string())

    phantom = Phantom("phantom3.xml")
    guidewire = Guidewire()
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
    )
    env = composer.Environment(
        task=task,
        strip_singleton_obs_buffer_dim=True,
        time_limit=5,
    )

    viewer.launch(env)

    # guidewire.mjcf_model.compiler.set_attributes(autolimits=True, angle="radian")
    # mjcf.Physics.from_mjcf_model(guidewire.mjcf_model)
    # guidewire.save_model(Path.cwd() / "guidewire_3.xml")
