from controlMotor.control import control
import random
import numpy as np

class real_env():
    __control = None
    __motorPort = "/dev/ttyUSB0"

    def __init__(self):
        self.__control = control(self.__motorPort)

    # def __del__(self):
    #     print("object destoryed")

    def reset(self):
        observation = self.__control.runTo(0.5, 0)
        return observation, {}

    def step(self, action):
        # action[-1 -- 1,-1 -- 1]
        [linear, rotate] = action
        observation = self.__control.run(linear, rotate)
        reward, terminated, truncated, info = None, None, None, None
        return observation, reward, terminated, truncated, info

    # def __readimg(self, imgfilename):
    #     cv2.
    #     for key, value in obs.items():
    #         if value.dtype == np.float64:
    #             obs[key] = value.astype(np.float32)
    #     mg = Image.open(filename)
    #     mats = np.array(mg)
    #     mg2 = Image.fromarray(mats)
