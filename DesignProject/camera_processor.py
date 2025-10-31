import cv2

class CameraProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise Exception("Unable to turn on camera")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release(self):
        self.cap.release()