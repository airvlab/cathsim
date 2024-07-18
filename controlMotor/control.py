#control each step have one image
from camera import camera 
from motor import motor 
import random

random.seed(42)

motor1=motor("/dev/ttyUSB0")
# camera1=camera()

def ramdomnumber():
	return random.uniform(-1,1)
def randomRun():
	for i in range (10):
		# filename=f"image_{i}.png"
		motor1.move(1, ramdomnumber(), ramdomnumber())
		# camera1.write_img(filename)
#romdom move and then gain the image

randomRun()
motor1.moveTo(1, 0.5, 0)