from cathsim.dm.components.base_models import BasePhantom, BaseGuidewire
from cathsim.gym.envs import CathSim
from dm_control import mjcf
from cathsim.dm import make_dm_env


class MyPhantom(BasePhantom):

    def _build(self):
        xml_string = """
        <mujoco>
        <worldbody>
            <body>
            <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
            </body>
        </worldbody>
        </mujoco>
        """
        self._mjcf_root = mjcf.from_xml_string(xml_string)

    def mjcf_model(self):
        return self._mjcf_root()


class MyGuidewire(BaseGuidewire):
    def _build(self):
        xml_string = """
        <mujoco>
        <worldbody>
            <body pos="0 -0.5">
            <geom type="capsule" size=".1 .1" rgba="0 .9 0 1"/>
            </body>
        </worldbody>
        </mujoco>
        """
        self._mjcf_root = mjcf.from_xml_string(xml_string)

    def mjcf_model(self):
        return self._mjcf_root()


if __name__ == "__main__":
    phantom = MyPhantom()
    guidewire = MyGuidewire()

    make_dm_env(phantom, guidewire)
