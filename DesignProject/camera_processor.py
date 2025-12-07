import pyrealsense2 as rs
import numpy as np
import cv2


class CameraProcessor:
    def __init__(self):
        # Check for connected RealSense devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise Exception("No Intel RealSense devices detected. Please connect your D435i camera.")
        
        # Print device info
        for dev in devices:
            print(f"Found RealSense device: {dev.get_info(rs.camera_info.name)}")
        
        # Create pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure color stream for D435i (1280x720, BGR8 format, 30fps)
        # D435i supports: 1920x1080@30, 1280x720@30/60, 640x480@30/60
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # Start streaming
        try:
            self.profile = self.pipeline.start(self.config)
            print("RealSense camera started successfully")
        except Exception as e:
            raise Exception(f"Unable to start RealSense camera: {e}\nMake sure your D435i is connected and not being used by another application.")
    
    def get_frame(self):
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                return frame
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def release(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"Error releasing camera: {e}")