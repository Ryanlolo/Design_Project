import cv2
import numpy as np
import sys
import os
try:
    import tensorflow as tf
    # Try different ways to import keras based on TensorFlow version
    if hasattr(tf, 'keras'):
        keras = tf.keras
    elif hasattr(tf, 'python') and hasattr(tf.python, 'keras'):
        keras = tf.python.keras
    else:
        # Try standalone keras
        try:
            import keras
        except ImportError:
            keras = None
except ImportError:
    keras = None

class PieceClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.use_cnn = False
        
        if model_path and os.path.exists(model_path) and keras is not None:
            try:
                self.model = keras.models.load_model(model_path)
                self.use_cnn = True
                print("CNN model loaded successfully!")
            except Exception as e:
                print(f"Failed to load CNN model: {e}")
                print("Falling back to color recognition methods")
                self.model = None
                self.use_cnn = False
        else:
            if model_path and not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
            if keras is None:
                print("Keras/TensorFlow not available")
            print("Using traditional color recognition methods")
    
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