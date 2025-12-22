import cv2
import numpy as np
import os
import sys
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


class BoardDetectorTrained:
    """
    Training-based board detector using deep learning model.
    Uses CNN to detect board corner points instead of traditional contour detection.
    """
    
    def __init__(self, model_path=None, input_size=(640, 480)):
        """
        Initialize the trained board detector.
        
        Args:
            model_path: Path to the trained model file (.h5)
            input_size: Input image size for the model (width, height)
        """
        self.model = None
        self.use_model = False
        self.input_size = input_size
        self.output_size = (300, 300)  # Output board size
        
        if model_path and os.path.exists(model_path) and keras is not None:
            try:
                self.model = keras.models.load_model(model_path)
                self.use_model = True
                print("Trained board detector model loaded successfully!")
            except Exception as e:
                print(f"Failed to load trained board detector model: {e}")
                print("Falling back to traditional detection methods")
                self.model = None
                self.use_model = False
        else:
            if model_path and not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
            if keras is None:
                print("Keras/TensorFlow not available")
            print("Using traditional contour-based detection methods")
    
    def detect_board(self, image):
        """
        Detect board in the image and return warped board image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            warped board image or None if board not detected
        """
        if self.use_model and self.model is not None:
            return self._detect_with_model(image)
        else:
            return self._detect_with_traditional(image)
    
    def _detect_with_model(self, image):
        """
        Detect board using trained CNN model.
        Model outputs 8 values (4 corner points: x1,y1,x2,y2,x3,y3,x4,y4)
        """
        # Preprocess image
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, self.input_size)
        normalized = resized.astype(np.float32) / 255.0
        input_batch = np.expand_dims(normalized, axis=0)
        
        # Predict corner points
        try:
            prediction = self.model.predict(input_batch, verbose=0)[0]
            
            # Extract 8 values (4 corners with x,y coordinates)
            # Normalize prediction to [0, 1] if needed
            corners = prediction.reshape(4, 2)
            
            # Scale corners back to original image size
            corners[:, 0] *= original_width
            corners[:, 1] *= original_height
            
            # Convert to integer coordinates
            corners = corners.astype(np.int32)
            
            # Perform perspective transform
            warped = self._perspective_transform(image, corners)
            return warped
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self._detect_with_traditional(image)
    
    def _detect_with_traditional(self, image):
        """
        Fallback to traditional contour-based detection.
        This is a simplified version of the original method.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive binarization
        binary = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular outline
        board_contour = None
        max_area = 0
        min_board_area = 50000
        
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it is a quadrilateral
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area and area > min_board_area:
                    max_area = area
                    board_contour = approx
        
        if board_contour is not None:
            # Perspective transformation
            corners = board_contour.reshape(4, 2)
            warped = self._perspective_transform(image, corners)
            return warped
        
        return None
    
    def _perspective_transform(self, image, corners):
        """
        Apply perspective transformation to get front view of board.
        
        Args:
            image: Original image
            corners: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Warped board image
        """
        # Rearrange corner points
        points = np.array(corners, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        
        # Calculate the sum and difference of points
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # upper left
        rect[2] = points[np.argmax(s)]  # lower right
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # upper right
        rect[3] = points[np.argmax(diff)]  # lower left
        
        # Define target points
        width, height = self.output_size
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Compute perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        return warped
    
    def draw_board_state(self, image, board_state):
        """
        Draw board state on the image.
        
        Args:
            image: Image to draw on
            board_state: 3x3 board state matrix
            
        Returns:
            Image with board state drawn
        """
        display_image = image.copy()
        
        # Show the board status on the screen
        for i in range(3):
            for j in range(3):
                status = board_state[i][j]
                x = 50 + j * 100
                y = 50 + i * 30
                
                if status == 'red':
                    color = (0, 0, 255)  # red
                    text = "red"
                elif status == 'blue':
                    color = (255, 0, 0)  # blue
                    text = "blue"
                else:
                    color = (128, 128, 128)  # grey
                    text = "null"
                
                cv2.putText(display_image, f"{text}({i},{j})", 
                          (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, color, 2)
        
        return display_image
    
    def draw_ai_suggestion(self, image, move):
        """
        Draw AI suggestion on the image.
        
        Args:
            image: Image to draw on
            move: Tuple (row, col) representing AI's suggested move
            
        Returns:
            Image with AI suggestion drawn
        """
        row, col = move
        display_image = image.copy()
        
        # Display AI suggestion as text with coordinates
        ai_text = f"AI suggests: ({row}, {col})"
        
        # Display at top-left corner
        cv2.putText(display_image, ai_text, (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return display_image

