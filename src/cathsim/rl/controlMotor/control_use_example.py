# control the motor randomRun run or run To
from control import control
import random

control = control("/dev/ttyUSB0")
for i in range(2):
    control.randomRun()

control.runTo(0.5, 0)

for i in range(2):
	control.run(1, 1)

control.runTo(0.5, 0)