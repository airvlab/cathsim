import mujoco
from mujoco import viewer
from dm_control import mjcf
import time

xml_model = """
<!-- Copyright 2021 DeepMind Technologies Limited

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
-->

<mujoco model="Cable">
  <include file="scene.xml"/>

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
    <composite type="cable" curve="s" count="41 1 1" size="1" offset="-.3 0 .6" initial="none">
      <plugin plugin="mujoco.elasticity.cable">
        <!--Units are in Pa (SI)-->
        <config key="twist" value="1e7"/>
        <config key="bend" value="4e6"/>
        <config key="vmax" value="0.05"/>
      </plugin>
      <joint kind="main" damping=".015"/>
      <geom type="capsule" size=".005" rgba=".8 .2 .1 1" condim="1"/>
    </composite>
    <body name="slider" pos=".7 0 .6">
      <joint type="slide" axis="1 0 0" damping=".1"/>
      <geom size=".01"/>
    </body>
  </worldbody>
  <equality>
    <connect name="right_boundary" body1="B_last" body2="slider" anchor=".025 0 0"/>
  </equality>
  <contact>
    <exclude body1="B_last" body2="slider"/>
  </contact>
  <actuator>
    <motor site="S_last" gear="0 0 0 1 0 0" ctrlrange="-.03 .03"/>
  </actuator>
</mujoco>
"""

if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_string(xml_model)
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
