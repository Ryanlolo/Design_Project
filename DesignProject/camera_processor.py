import pyrealsense2 as rs
import numpy as np
import cv2


class CameraProcessor:
    def __init__(self, vertical_offset: int = 0, horizontal_offset: int = 0):
        """
        vertical_offset: pixels to shift the view.
            > 0: move view upward, pad bottom
            < 0: move view downward, pad top
            = 0: no shift
        horizontal_offset: pixels to shift horizontally.
            > 0: move view to the right (crop left, pad right)
            < 0: move view to the left (crop right, pad left)
            = 0: no shift
        """
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
        
        # Cache frame dimensions for cropping
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.frame_width = color_profile.width()
        self.frame_height = color_profile.height()
        self.vertical_offset = int(vertical_offset)
        self.horizontal_offset = int(horizontal_offset)
    
    def get_frame(self):
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Shift view vertically by cropping and padding
                if self.vertical_offset > 0:
                    offset = min(self.vertical_offset, frame.shape[0] - 1)
                    shifted = frame[offset:, :]
                    pad_rows = frame.shape[0] - shifted.shape[0]
                    if pad_rows > 0:
                        padding = np.zeros((pad_rows, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
                        frame = np.vstack((shifted, padding))
                    else:
                        frame = shifted
                elif self.vertical_offset < 0:
                    offset = min(abs(self.vertical_offset), frame.shape[0] - 1)
                    shifted = frame[: frame.shape[0] - offset, :]
                    pad_rows = frame.shape[0] - shifted.shape[0]
                    if pad_rows > 0:
                        padding = np.zeros((pad_rows, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
                        frame = np.vstack((padding, shifted))
                    else:
                        frame = shifted
                
                # Shift view horizontally by cropping and padding
                if self.horizontal_offset > 0:
                    offset = min(self.horizontal_offset, frame.shape[1] - 1)
                    shifted = frame[:, offset:, :]
                    pad_cols = frame.shape[1] - shifted.shape[1]
                    if pad_cols > 0:
                        padding = np.zeros((frame.shape[0], pad_cols, frame.shape[2]), dtype=frame.dtype)
                        frame = np.hstack((shifted, padding))
                    else:
                        frame = shifted
                elif self.horizontal_offset < 0:
                    offset = min(abs(self.horizontal_offset), frame.shape[1] - 1)
                    shifted = frame[:, : frame.shape[1] - offset, :]
                    pad_cols = frame.shape[1] - shifted.shape[1]
                    if pad_cols > 0:
                        padding = np.zeros((frame.shape[0], pad_cols, frame.shape[2]), dtype=frame.dtype)
                        frame = np.hstack((padding, shifted))
                    else:
                        frame = shifted
                
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