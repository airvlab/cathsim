import mujoco
from mujoco import viewer


model = mujoco.MjModel.from_xml_path('/home/tuanvo1/Documents/AI_Recidence/cathsim/cathsim/cathsim/assets/phantom3_just_blood_and_aorta.xml')
viewer.launch(model=model)


