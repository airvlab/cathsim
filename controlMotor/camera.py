import pyrealsense2 as rs
import numpy as np
import cv2

class camera():
    __pipeline=None
    __images=None
    def __init__(self):
        # Configure depth and color streams
        self.__pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.__pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.__pipeline.start(config)
        # Wait for a coherent pair of frames: depth and color
        while True:
            frames = self.__pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                break
        # Convert images to numpy arrays

        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape

        self.__images=color_image
        
        
    def __del__(self):
        # Stop streaming
        self.__pipeline.stop()
        
    def write_img(self,filename):  
        # save image
        cv2.imwrite( filename, self.__images)
 
#  # use example 
# test=camera()
# test.write_img('image2.png')
