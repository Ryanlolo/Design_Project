# collect_training_data.py
import cv2
import os
import numpy as np

class CameraProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        # Set lower resolution to improve performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
        print(f"Camera {camera_index} opened successfully")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

class BoardDetector:
    def __init__(self):
        self.min_board_area = 10000  # Lower area requirement

    def detect_board(self, image):
        if image is None:
            return None
            
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
        
        # Find the largest rectangular contour
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
        try:
            # Rearrange contour points
            points = contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            rect[2] = points[np.argmax(s)]
            
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            rect[3] = points[np.argmax(diff)]
            
            # Define target points
            width = height = 300
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
        except Exception as e:
            print(f"Perspective transformation error: {e}")
            return None

    def _split_cells(self, board_image):
        """Split board into 3x3 grid"""
        if board_image is None:
            return []
            
        height, width = board_image.shape[:2]
        cell_height = height // 3
        cell_width = width // 3
        
        cells = []
        for i in range(3):
            for j in range(3):
                y_start = i * cell_height + 5  # Reduce edge buffer
                y_end = (i + 1) * cell_height - 5
                x_start = j * cell_width + 5
                x_end = (j + 1) * cell_width - 5
                
                cell = board_image[y_start:y_end, x_start:x_end]
                if cell.size > 0:
                    cells.append(cell)
        
        return cells

def create_training_folders():
    """Create training folders"""
    folders = ['empty', 'red', 'blue']
    for folder in folders:
        os.makedirs(f'training_data/{folder}', exist_ok=True)
    print("Training folder structure created")

def save_cells(cells, label, counter):
    """Save cell images"""
    saved_count = 0
    for i, cell in enumerate(cells):
        if cell is not None and cell.size > 0:
            filename = f'training_data/{label}/{label}_{counter:04d}_{i}.jpg'
            try:
                cv2.imwrite(filename, cell)
                saved_count += 1
            except Exception as e:
                print(f"Failed to save image: {e}")
    
    counter += 1
    print(f"Saved {saved_count} images as {label}")
    return counter

def collect_training_data():
    """Main data collection function"""
    
    # Create folders
    create_training_folders()
    
    # Try different camera indices
    camera_index = 0
    camera = None
    
    for i in range(3):  # Try 0, 1, 2
        try:
            camera = CameraProcessor(camera_index=i)
            print(f"Successfully using camera index: {i}")
            break
        except Exception as e:
            print(f"Camera {i} cannot be opened: {e}")
            continue
    
    if camera is None:
        print("Cannot open any camera, please check connection")
        return

    detector = BoardDetector()
    
    print("\n=== Data Collection Mode ===")
    print("Press 'e' to save as empty")
    print("Press 'r' to save as red") 
    print("Press 'b' to save as blue")
    print("Press 'c' to show/hide cell preview")
    print("Press 'q' to quit")
    print("===================\n")
    
    counter = {'empty': 0, 'red': 0, 'blue': 0}
    show_cells = False
    
    try:
        while True:
            # Read camera frame
            frame = camera.get_frame()
            if frame is None:
                print("Cannot read camera frame")
                break
            
            # Detect board
            board_roi = detector.detect_board(frame)
            
            display_frame = frame.copy()
            cells = []
            
            if board_roi is not None:
                # Mark board on screen
                cv2.putText(display_frame, "Board Detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Split cells
                cells = detector._split_cells(board_roi)
                
                if show_cells and cells:
                    # Show cell preview
                    for i, cell in enumerate(cells):
                        if cell is not None and cell.size > 0:
                            row = i // 3
                            col = i % 3
                            # Show small preview on main screen
                            preview_x = 10 + col * 70
                            preview_y = 50 + row * 70
                            
                            # Resize preview
                            preview = cv2.resize(cell, (60, 60))
                            display_frame[preview_y:preview_y+60, preview_x:preview_x+60] = preview
                            
                            # Mark cell number
                            cv2.putText(display_frame, str(i), (preview_x, preview_y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(display_frame, "No Board Detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display counter information
            info_text = f"Empty: {counter['empty']} | Red: {counter['red']} | Blue: {counter['blue']}"
            cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Collect Data - Press Q to quit', display_frame)
            
            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e') and cells:  # Save as empty
                counter['empty'] = save_cells(cells, 'empty', counter['empty'])
            elif key == ord('r') and cells:  # Save as red
                counter['red'] = save_cells(cells, 'red', counter['red'])
            elif key == ord('b') and cells:  # Save as blue
                counter['blue'] = save_cells(cells, 'blue', counter['blue'])
            elif key == ord('c'):  # Toggle cell preview
                show_cells = not show_cells
                print(f"Cell preview: {'On' if show_cells else 'Off'}")
    
    except Exception as e:
        print(f"Program execution error: {e}")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\nData collection completed! Total:")
        print(f"Empty: {counter['empty']} sets")
        print(f"Red: {counter['red']} sets") 
        print(f"Blue: {counter['blue']} sets")

def simple_collect_mode():
    """Simplified data collection mode without board detection"""
    
    create_training_folders()
    
    try:
        camera = CameraProcessor(0)
    except:
        print("Cannot open camera")
        return
    
    print("\n=== Simplified Data Collection Mode ===")
    print("Capture entire frame directly")
    print("Press 'e', 'r', 'b' to save current frame")
    print("Press 'q' to quit")
    
    counter = {'empty': 0, 'red': 0, 'blue': 0}
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            # Display simple frame
            display_frame = frame.copy()
            
            # Display instructions
            cv2.putText(display_frame, "Simple Collection Mode", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press E=Empty, R=Red, B=Blue, Q=Quit", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_text = f"Empty: {counter['empty']} | Red: {counter['red']} | Blue: {counter['blue']}"
            cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Simple Collection Mode', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):  # Save as empty
                filename = f'training_data/empty/empty_{counter["empty"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['empty'] += 1
                print(f"Saved empty image: {filename}")
            elif key == ord('r'):  # Save as red
                filename = f'training_data/red/red_{counter["red"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['red'] += 1
                print(f"Saved red image: {filename}")
            elif key == ord('b'):  # Save as blue
                filename = f'training_data/blue/blue_{counter["blue"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['blue'] += 1
                print(f"Saved blue image: {filename}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select data collection mode:")
    print("1. Automatic board detection mode (recommended)")
    print("2. Simplified mode (if automatic mode has issues)")
    
    choice = input("Please select (1/2): ").strip()
    
    if choice == "1":
        collect_training_data()
    elif choice == "2":
        simple_collect_mode()
    else:
        print("Invalid choice, using automatic mode")
        collect_training_data()