import math
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

xml_path = Path.cwd() / "src/cathsim/components/scene.xml"

m = mujoco.MjModel.from_xml_path(xml_path.as_posix())
d = mujoco.MjData(m)

translation_step = 0.002
rotation_step = 15
max_iters = 100
tolerance = 1e-3


def key_callback(keycode):
    if keycode == 265:
        d.ctrl[0] = d.ctrl[0] + translation_step
    elif keycode == 264:
        d.ctrl[0] = d.ctrl[0] - translation_step
    elif keycode == 263:
        d.ctrl[1] = d.ctrl[1] - math.radians(rotation_step)
    elif keycode == 262:
        d.ctrl[1] = d.ctrl[1] + math.radians(rotation_step)


with viewer.launch_passive(m, d, key_callback=key_callback) as mj_viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    time_elapsed = time.time() - start
    print("time_elapsed", time_elapsed)
    while mj_viewer.is_running() and time_elapsed < 30:
        step_start = time.time()

        translation_joint = d.joint("slider").qpos[0]
        rotation_joint = d.joint("rotator").qpos[0]

        d.ctrl = [
            translation_joint + translation_step,
            rotation_joint + math.radians(rotation_step),
        ]

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        for _ in range(max_iters):
            mujoco.mj_step(m, d)
            translation_joint = d.joint("slider").qpos[0]
            rotation_joint = d.joint("rotator").qpos[0]
            mj_viewer.sync()
            if np.all(np.abs([translation_joint - d.ctrl[0], rotation_joint - d.ctrl[1]]) < tolerance):
                break

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        mj_viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
