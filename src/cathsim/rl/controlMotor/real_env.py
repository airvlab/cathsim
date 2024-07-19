from control import control
import random

control = control("/dev/ttyUSB0")
for i in range(2):
    control.randomRun()


class real_env():
    __control = None
    __motorPort = "/dev/ttyUSB0"

    def __init__(self):
        self.__control = control(self.__motorPort)

    # def __del__(self):
    #     print("object destoryed")

    def reset(self):
        self.__control.runTo(0.5, 0)
        return self.__control

    def step(self, action):
        linear, rotate=action
        self.__control.run(linear, rotate)
