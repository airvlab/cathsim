from time import sleep
import serial


class Controller:
    _port=None
    _ser = None
    _curPosition = 0  # each step is 5mm ,so every move need to times 5
    _rightBound = 30000  # the right bound of step position is 0
    _leftBound = -30000  # the left bound of step position is -6000

    def __init__(self, port):
        self._port=port
        # open usb0 and set brut 115200
        self._ser = serial.Serial(self._port, 115200)

    def __del__(self):
        # for next time have the same global position
        self._move_to_global_position(0,0)
        self._ser.close()

    def send(self, enable, motor1, motor2, motor3, motor4, relative):
        data = bytearray(19)
        if enable:
            data[0] = 0x81
        else:
            data[0] = 0x80
        if relative:
            data[18] = 0x80
        else:
            data[18] = 0x81
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
        self._ser.write(data)
        self._ser.flush()
        sleep(2)

    def _check_bound(self, check_position):
        assert (
            self._leftBound <= check_position <= self._rightBound
        ), f"The move out of range, and be cancelled"
        return True

    def _check_type_range(self, translation, rotation):
        # do the checks:type and range check
        assert isinstance(
            translation, float
        ), f"Got translation {type(translation)}, expected float"
        assert isinstance(
            rotation, float
        ), f"Got rotation {type(rotation)}, expected float"
        assert (
            -1 <= translation <= 1
        ), f"Got translation {translation}, expected in range (-1 1)"
        assert (
            -1 <= rotation <= 1
        ), f"Got rotation {rotation}, expected in range (-1, 1)"
        return True

    def get_inf(self):
        return (
            self._curPosition / 100.0,
            self._rightBound / 100.0,
            self._leftBound / 100.0,
        )

    def move(self, translation, rotation, relative=True):
        if self._check_type_range(translation, rotation):
            if relative:
                self._move_to_relative_position(
                    translation=translation, rotation=rotation
                )
            else:
                self._move_to_global_position(
                    translation=translation, rotation=rotation
                )

    def _move_to_relative_position(self, translation, rotation):
        # motor3B, motor4B should be in range(-1,1)
        motor3_scale_factor = 500  # 5 mm; 800 step one rotation -8mm
        motor4_scale_factor = 200  # 90 degree; 800 step 360 degree

        motor3 = int(translation * float(motor3_scale_factor))
        motor4 = int(rotation * float(motor4_scale_factor))
        expectPosition = self._curPosition + motor3
        if self._check_bound(check_position=expectPosition):
            self.send(
                enable=True,
                motor1=0,
                motor2=0,
                motor3=motor3,
                motor4=motor4,
                relative=True,
            )

    def _move_to_global_position(self, translation, rotation):
        # motor3B, motor4B should be in range(0,1)
        # change range from(-1,1) to range (0,1)
        # translation = (translation + 1.0) / 2.0
        # rotation = (rotation + 1.0) / 2.0

        motor3_scale_factor = -30000  # total 600 mm; 800 step one rotation -8mm
        motor4_scale_factor = 800  # 360 degree;

        motor3 = int(translation * float(motor3_scale_factor))
        motor4 = int(rotation * float(motor4_scale_factor))
        self.send(
            enable=True,
            motor1=0,
            motor2=0,
            motor3=motor3,
            motor4=motor4,
            relative=False,
        )



