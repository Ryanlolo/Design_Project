import cv2
import numpy as np

class BoardDetector:
    def __init__(self):
        self.min_board_area = 50000  # Minimum chessboard area
    
    def detect_board(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # adaptive binarization
        binary = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular outline
        board_contour = None
        max_area = 0
        
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it is a quadrilateral
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area and area > self.min_board_area:
                    max_area = area
                    board_contour = approx
        
        if board_contour is not None:
            # Perspective transformation to obtain front view
            warped = self._perspective_transform(image, board_contour)
            return warped
        
        return None
    
    def _perspective_transform(self, image, contour):
        # Rearrange contour points
        points = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Calculate the sum and difference of points
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # upper left
        rect[2] = points[np.argmax(s)]  # lower right
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # upper right
        rect[3] = points[np.argmax(diff)]  # lower left
        
        # Define target point
        width = 300
        height = 300
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
        row, col = move
        display_image = image.copy()
        
        # Display AI suggestion as text with coordinates
        ai_text = f"AI suggests: ({row}, {col})"
        
        # Display at top-left corner
        cv2.putText(display_image, ai_text, (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return display_image