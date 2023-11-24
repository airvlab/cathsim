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


if __name__ == "__main__":
    from cathsim.dm import make_dm_env, Navigate
    from cathsim.dm.components import Phantom, Guidewire, Tip
    from dm_control import viewer
    guidewire = Guidewire()
    # print(guidewire._mjcf_root.to_xml_string())

    phantom = Phantom("phantom3.xml")
    guidewire = Guidewire()
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=Tip(),
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
