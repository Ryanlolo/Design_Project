import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Check GPU availability for inference
if tf.config.list_physical_devices('GPU'):
    print("[OK] Using GPU for piece classification!")
else:
    print("[INFO] Using CPU for piece classification")

class PieceClassifier:
    def __init__(self, model_path=None):
        if model_path and tf.io.gfile.exists(model_path):
            self.model = keras.models.load_model(model_path)
            self.use_cnn = True
        else:
            self.model = None
            self.use_cnn = False
            print("Use traditional color recognition methods")
    
    def analyze_board(self, board_image):
        # Divide the chessboard into a 3x3 grid
        cells = self._split_cells(board_image)
        board_state = [['empty' for _ in range(3)] for _ in range(3)]
        
        for i, cell in enumerate(cells):
            row = i // 3
            col = i % 3
            
            if self.use_cnn:
                # Recognition using CNN model
                board_state[row][col] = self._classify_with_cnn(cell)
            else:
                # Use color identification
                board_state[row][col] = self._classify_with_color(cell)
        
        return board_state
    
    def _split_cells(self, board_image):
        height, width = board_image.shape[:2]
        cell_height = height // 3
        cell_width = width // 3
        
        cells = []
        for i in range(3):
            for j in range(3):
                y_start = i * cell_height + 10  # edge buffer
                y_end = (i + 1) * cell_height - 10
                x_start = j * cell_width + 10
                x_end = (j + 1) * cell_width - 10
                
                cell = board_image[y_start:y_end, x_start:x_end]
                cells.append(cell)
        
        return cells
    
    def _classify_with_color(self, cell_image):
        if cell_image.size == 0:
            return 'empty'
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
        
        # Define color range
        # Red range (note that the Hue range in OpenCV is 0-180)
        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])
        
        # blue range
        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([140, 255, 255])
        
        # Create a mask
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Count the number of pixels
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Set threshold
        threshold = cell_image.shape[0] * cell_image.shape[1] * 0.1
        
        if red_pixels > threshold and red_pixels > blue_pixels:
            return 'red'
        elif blue_pixels > threshold and blue_pixels > red_pixels:
            return 'blue'
        else:
            return 'empty'
    
    def _classify_with_cnn(self, cell_image):
        # Preprocess images
        processed = cv2.resize(cell_image, (64, 64))
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        # predict
        prediction = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        
        classes = ['empty', 'red', 'blue']
        return classes[class_idx]