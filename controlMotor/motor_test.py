# control each step have one image
from camera import camera
from motor import motor
import random

seed = 42

# port:"/dev/ttyUSB0"
# romdom move and then gain the image
motorPort = "/dev/ttyUSB0"
motor = motor(motorPort)
random.seed(seed)

linear, rotate = random.uniform(-10, 10), random.uniform(-1, 1)
motor.move(1, linear, rotate)
print(linear, rotate)