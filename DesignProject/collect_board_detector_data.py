"""
Data collection script for board detector training.
This script allows you to manually annotate board corner points in images.
"""
import cv2
import numpy as np
import os
import json
import sys

# Try to import camera processor, fallback to OpenCV if not available
try:
    from camera_processor import CameraProcessor
    USE_REALSENSE = True
except ImportError:
    USE_REALSENSE = False
    print("RealSense camera not available, using OpenCV camera instead")


class SimpleCameraProcessor:
    """Simple camera processor using OpenCV (fallback)"""
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
        print(f"Camera {camera_index} opened successfully")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()


class BoardAnnotationTool:
    """Interactive tool for annotating board corner points"""
    
    def __init__(self):
        self.points = []  # List of 4 corner points
        self.current_image = None
        self.display_image = None
        self.window_name = "Board Annotation Tool"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for clicking corner points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"Point {len(self.points)}: ({x}, {y})")
                self._update_display()
            else:
                print("All 4 points have been selected. Press 's' to save or 'c' to clear.")
    
    def _update_display(self):
        """Update the display image with current annotations"""
        self.display_image = self.current_image.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            x, y = point
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"P{i+1}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw lines connecting points (if we have 2 or more points)
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                pt1 = tuple(self.points[i])
                pt2 = tuple(self.points[i + 1])
                cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
            
            # Close the polygon if we have 4 points
            if len(self.points) == 4:
                cv2.line(self.display_image, 
                        tuple(self.points[3]), 
                        tuple(self.points[0]), 
                        (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Click 4 corner points of the board (clockwise or counter-clockwise)",
            f"Points selected: {len(self.points)}/4",
            "Press 's' to save, 'c' to clear, 'q' to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def annotate_image(self, image):
        """Annotate corner points on an image"""
        self.current_image = image.copy()
        self.points = []
        self._update_display()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return None  # Quit without saving
            elif key == ord('c'):
                self.points = []
                self._update_display()
                print("Points cleared")
            elif key == ord('s'):
                if len(self.points) == 4:
                    return self.points.copy()
                else:
                    print("Please select exactly 4 points before saving")
        
        cv2.destroyWindow(self.window_name)
        return None


def load_or_create_annotations(annotations_file):
    """Load existing annotations or create empty dict"""
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            return json.load(f)
    return {}


def save_annotations(annotations, annotations_file):
    """Save annotations to JSON file"""
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotations saved to {annotations_file}")


def collect_from_camera(data_dir='training_data/board_detector'):
    """Collect training data from camera"""
    
    # Create directories
    images_dir = os.path.join(data_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    annotations_file = os.path.join(data_dir, 'annotations.json')
    annotations = load_or_create_annotations(annotations_file)
    
    # Initialize camera
    if USE_REALSENSE:
        try:
            camera = CameraProcessor()
        except Exception as e:
            print(f"RealSense camera failed: {e}")
            print("Falling back to OpenCV camera")
            camera = SimpleCameraProcessor()
    else:
        camera = SimpleCameraProcessor()
    
    annotation_tool = BoardAnnotationTool()
    
    print("\n=== Board Detector Data Collection ===")
    print("Instructions:")
    print("1. Position the board in the camera view")
    print("2. Press SPACE to capture the current frame")
    print("3. Click 4 corner points of the board in order")
    print("4. Press 's' to save annotation, 'c' to clear, 'q' to skip")
    print("5. Press 'q' in main window to quit")
    print("=====================================\n")
    
    image_counter = len(annotations)
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            # Display instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to capture frame", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Collected: {image_counter} images", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera - Press SPACE to capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar to capture
                # Annotate the captured frame
                points = annotation_tool.annotate_image(frame)
                
                if points is not None:
                    # Save image
                    image_filename = f"board_{image_counter:04d}.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    cv2.imwrite(image_path, frame)
                    
                    # Save annotation
                    annotations[image_filename] = points
                    save_annotations(annotations, annotations_file)
                    
                    print(f"Saved: {image_filename} with {len(points)} points")
                    image_counter += 1
                else:
                    print("Annotation cancelled")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\nData collection completed!")
        print(f"Total images collected: {image_counter}")
        save_annotations(annotations, annotations_file)


def annotate_existing_images(data_dir='training_data/board_detector'):
    """Annotate existing images in a directory"""
    
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    annotations_file = os.path.join(data_dir, 'annotations.json')
    annotations = load_or_create_annotations(annotations_file)
    
    annotation_tool = BoardAnnotationTool()
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    print(f"\nFound {len(image_files)} images")
    print("Press 's' to save, 'c' to clear, 'q' to skip/quit")
    print("After annotating all images, press 'q' to quit\n")
    
    for img_file in image_files:
        # Skip if already annotated
        if img_file in annotations:
            print(f"Skipping {img_file} (already annotated)")
            continue
        
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Could not load {img_file}")
            continue
        
        print(f"\nAnnotating: {img_file}")
        points = annotation_tool.annotate_image(image)
        
        if points is not None:
            annotations[img_file] = points
            save_annotations(annotations, annotations_file)
            print(f"Saved annotation for {img_file}")
        else:
            print(f"Skipped {img_file}")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                break
    
    cv2.destroyAllWindows()
    save_annotations(annotations, annotations_file)
    print(f"\nAnnotation completed! Total annotated: {len(annotations)}")


if __name__ == "__main__":
    print("Board Detector Data Collection Tool")
    print("=" * 40)
    print("1. Collect data from camera")
    print("2. Annotate existing images")
    print("=" * 40)
    
    choice = input("Select option (1/2): ").strip()
    
    if choice == "1":
        collect_from_camera()
    elif choice == "2":
        annotate_existing_images()
    else:
        print("Invalid choice, using camera collection")
        collect_from_camera()

