import pyrealsense2 as rs
import numpy as np

import cv2

def Colour2Grayscale(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ExposureLock(pline):
    cam = pline.start().get_device().query_sensors()[1]

    cam.set_option(rs.option.exposure, 10)

    return cam, pline

if __name__ == "__main__":
    cam_frames = np.array([])
    pline = rs.pipeline()
    cam, pline = ExposureLock(pline)

    try:
        for i in range(100):
            frames = pline.wait_for_frames()
            rgb_frame = frames.get_color_frame()
            rgb_frame = np.asanyarray(rgb_frame.get_data())

            gray_frame = Colour2Grayscale(rgb_frame)
            print(gray_frame.shape)

            cam_frames[]
    finally:
        cam.stop()