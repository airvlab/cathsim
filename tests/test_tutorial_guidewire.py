from cathsim.dm.components.base_models import BaseGuidewire
from dm_control import mjcf
import mujoco

xml_string = """
<mujoco model="Cable">

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <statistic center="0 0 .3" extent="1"/>
  <visual>
    <global elevation="-30"/>
  </visual>

  <compiler autolimits="true"/>

  <size memory="2M"/>

  <worldbody>
    <composite type="cable" curve="s" count="80 1 1" size="0.25" offset="0.000025 0 0" initial="none">
      <plugin plugin="mujoco.elasticity.cable">
        <!--Units are in Pa (SI)-->
        <config key="twist" value="1e7"/>
        <config key="bend" value="1e9"/>
        <config key="vmax" value="0.05"/>
      </plugin>
      <joint kind="main" damping=".015"/>
      <geom type="capsule" size=".001" rgba=".8 .2 .1 1" condim="1"/>
    </composite>
    <body name="slider" pos=".7 0 .6">
      <joint type="slide" axis="1 0 0" damping=".1"/>
      <geom size=".01"/>
    </body>
  </worldbody>
  <actuator>
    <motor site="S_last" gear="0 0 0 1 0 0" ctrlrange="-.03 .03"/>
  </actuator>
</mujoco>
"""


class MyGuidewire(BaseGuidewire):

    def _build(self):
        self._mjcf_root = mjcf.from_xml_string(xml_string)

    def mjcf_model(self):
        return self._mjcf_root


if __name__ == "__main__":
    import time
    import mujoco.viewer

    MyGuidewire()

    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
