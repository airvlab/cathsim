import serial
import serial.tools.list_ports

# # to get the port
# ports_list = list(serial.tools.list_ports.comports())
# if len(ports_list) <= 0:
#     print("no port")
# else:
#     print("available port as followings：")
#     for comport in ports_list:
#         print(list(comport)[0], list(comport)[1])
# print(ser.name)  # get the port name


def send(enable, motor1, motor2, motor3, motor4):
    data = bytearray(18)
    if enable:
        data[0] = 0x81
    else:
        data[0] = 0x80
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
    # print(data)  # To verify the output
    ser.write(data)
    ser.flush()
    print("Finished the data trans")  # To verify the output


def move(enable, motor3B, motor4B):
    motor3 = 1000  # 10 mm; 800 step one rotation -8mm
    motor4 = 22  # 9.9 degree; 800 step 360 degree
    if motor3B == 0:
        motor3 = -motor3
    if motor4B == 0:
        motor4 = -motor4
    send(enable, 0, 0, motor3, motor4)


ser = serial.Serial("/dev/ttyUSB0", 115200)  # 打开COM17，将波特率配置为115200，其余参数使用默认值
if ser.isOpen():  # 判断串口是否成功打开
    print("open")
    # send(1, 1, 2, 4000, 400)
    move(1, 0, 0)
else:
    print("failed to open")
