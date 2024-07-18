from time import sleep
import serial
import serial.tools.list_ports


class motor():
    __ser = None

    def __init__(self, port):
        # open usb0 and set brut 115200
        self.__ser = serial.Serial(port, 115200)

    def __del__(self):
        self.__ser.close()

    def __send(self, enable, motor1, motor2, motor3, motor4, shouldMoveTo):
        data = bytearray(19)
        if enable:
            data[0] = 0x81
        else:
            data[0] = 0x80
        if shouldMoveTo:
            data[18] = 0x81
        else:
            data[18] = 0x80
        data[2] = (motor1 & 0xFF000000) >> 24
        data[3] = (motor1 & 0x00FF0000) >> 16
        data[4] = (motor1 & 0x0000FF00) >> 8
        data[5] = motor1 & 0x000000FF
        data[6] = (motor2 & 0xFF000000) >> 24
        data[7] = (motor2 & 0x00FF0000) >> 16
        data[8] = (motor2 & 0x0000FF00) >> 8
        data[9] = motor2 & 0x000000FF
        data[10] = (motor3 & 0xFF000000) >> 24
        data[11] = (motor3 & 0x00FF0000) >> 16
        data[12] = (motor3 & 0x0000FF00) >> 8
        data[13] = motor3 & 0x000000FF
        data[14] = (motor4 & 0xFF000000) >> 24
        data[15] = (motor4 & 0x00FF0000) >> 16
        data[16] = (motor4 & 0x0000FF00) >> 8
        data[17] = motor4 & 0x000000FF
        data[1] = 0x88  # setting this here as per the original C++ code
        self.__ser.write(data)
        self.__ser.flush()
        sleep(5)

    def move(self, enable, motor3B, motor4B):
        # motor3B, motor4B should be in range(-1,1)
        motor3_step = 500  # 5 mm; 800 step one rotation -8mm
        motor4_step = 200  # 90 degree; 800 step 360 degree
        motor3 = int(motor3B*float(motor3_step))
        motor4 = int(motor4B*float(motor4_step))
        self.__send(enable, 0, 0, motor3, motor4, False)
        self.__send(enable, 0, 0, motor3, motor4, False)

    def moveTo(self, enable, motor3B, motor4B):
        # motor3B, motor4B should be in range(0,1)
        motor3_step = 60000  # 5 mm; 800 step one rotation -8mm
        motor4_step = 800  # 90 degree; 800 step 360 degree
        motor3 = int(motor3B*float(motor3_step))
        motor4 = int(motor4B*float(motor4_step))
        self.__send(enable, 0, 0, motor3, motor4, True)
        self.__send(enable, 0, 0, motor3, motor4, True)
