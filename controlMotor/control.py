# control each step have one image
from camera import camera
from motor import motor
import random

# port:"/dev/ttyUSB0"
# romdom move and then gain the image


class control():
    __i = 0
    __seed = 42
    __motorPort = None
    __motor = None
    __camera = None

    # def __init__(self):
    #     self.__camera = camera()
    #     random.seed(self.__seed)

    def __init__(self, motorPort):
        self.__motorPort = motorPort
        self.__motor = motor(self.__motorPort)
        self.__camera = camera()
        random.seed(self.__seed)

    # def __del__(self):
    #     print("Object destroyed")

    def __ramdomnumber(self):
        if (random.randint(0, 1)):
            return 0, random.uniform(-1, 1)
        else:
            return random.uniform(-1, 1), 0

    def reset_seed(self, seed):
        self.__seed = seed
        random.seed(self.__seed)

    def setMotorPort(self, motorPort):
        self.__motorPort = motorPort
        self.__motor = (self.__motorPort)

    def randomRun(self):
        if self.__motorPort==None:
            print("Haven't give the port of motor! Please use \"setMotorPort\" to give the port ")
            return
        filename = f"image_{self.__i}.png"
        self.__i = self.__i+1
        linear, rotate = self.__ramdomnumber()
        self.__motor.move(1, linear, rotate)
        self.__camera.write_img(filename)

    def run(self, linear, rotate):
        if self.__motorPort==None:
            print("Haven't give the port of motor! Please use \"setMotorPort\" to give the port ")
            return
        filename = f"image_{self.__i}.png"
        self.__i = self.__i+1
        self.__motor.move(1, linear, rotate)
        self.__camera.write_img(filename)
    def runTo(self, linear, rotate):
        if self.__motorPort==None:
            print("Haven't give the port of motor! Please use \"setMotorPort\" to give the port ")
            return
        filename = f"image_{self.__i}.png"
        self.__i = self.__i+1
        self.__motor.moveTo(1, linear, rotate)
        self.__camera.write_img(filename)