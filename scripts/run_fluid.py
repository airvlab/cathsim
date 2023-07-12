import mujoco
from mujoco import viewer


model = mujoco.MjModel.from_xml_path('../cathsim/cathsim/cathsim/assets/phantom3_just_fluid.xml')
viewer.launch(model=model)

