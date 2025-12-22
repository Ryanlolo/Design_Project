"""
Training script for board detector model.
This model learns to predict 4 corner points of the chessboard (8 values total).
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import json
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Check for GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("[OK] Using GPU for training!")
else:
    print("[WARN] Using CPU for training (GPU not detected)")


def create_board_detector_model(input_size=(640, 480)):
    """
    Create CNN model for board detection.
    Model outputs 8 values representing 4 corner points (x1,y1,x2,y2,x3,y3,x4,y4).
    
    Args:
        input_size: Tuple (width, height) for input image size
        
    Returns:
        Compiled Keras model
    """
    width, height = input_size
    
    # Use a more sophisticated architecture for object detection
    inputs = keras.Input(shape=(height, width, 3))
    
    # Feature extraction layers
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    
    # Flatten and dense layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Output layer: 8 values for 4 corner points (normalized coordinates [0, 1])
    outputs = keras.layers.Dense(8, activation='sigmoid', name='corner_points')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model


def load_training_data(data_dir, input_size=(640, 480)):
    """
    Load training data for board detector.
    
    Expected data structure:
    training_data/board_detector/
        images/
            image1.jpg
            image2.jpg
            ...
        annotations.json  # JSON file with corner point annotations
    
    JSON format:
    {
        "image1.jpg": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "image2.jpg": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        ...
    }
    
    Args:
        data_dir: Directory containing training data
        input_size: Input image size for model
        
    Returns:
        Tuple (images, labels) as numpy arrays
    """
    images = []
    labels = []
    
    images_dir = os.path.join(data_dir, 'images')
    annotations_file = os.path.join(data_dir, 'annotations.json')
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return np.array([]), np.array([])
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        print("Please create annotations.json with corner point labels")
        return np.array([]), np.array([])
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    width, height = input_size
    
    # Load images and corresponding labels
    for img_file, corners in annotations.items():
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        original_height, original_width = img.shape[:2]
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0
        
        images.append(normalized)
        
        # Normalize corner coordinates to [0, 1]
        normalized_corners = []
        for x, y in corners:
            norm_x = x / original_width
            norm_y = y / original_height
            normalized_corners.extend([norm_x, norm_y])
        
        labels.append(normalized_corners)
    
    return np.array(images), np.array(labels)


def create_sample_annotation_template(data_dir):
    """
    Create a sample annotation template file.
    """
    template = {
        "example_image.jpg": [
            [100, 100],  # top-left corner
            [500, 100],  # top-right corner
            [500, 400],  # bottom-right corner
            [100, 400]   # bottom-left corner
        ]
    }
    
    annotations_file = os.path.join(data_dir, 'annotations_template.json')
    with open(annotations_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created annotation template at: {annotations_file}")
    print("Please rename it to 'annotations.json' and fill in your data")


def train(data_dir='training_data/board_detector', 
          input_size=(640, 480),
          epochs=50,
          batch_size=8,
          model_save_path='models/board_detector.h5'):
    """
    Train the board detector model.
    
    Args:
        data_dir: Directory containing training data
        input_size: Input image size (width, height)
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Path to save trained model
    """
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_training_data(data_dir, input_size)
    
    if len(X_train) == 0:
        print("No training data found!")
        print(f"Expected structure:")
        print(f"  {data_dir}/")
        print(f"    images/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"      ...")
        print(f"    annotations.json")
        print("\nCreating sample annotation template...")
        create_sample_annotation_template(data_dir)
        return
    
    print(f"Loaded {len(X_train)} training samples")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Label shape: {y_train[0].shape}")
    
    # Create model
    print("\nCreating model...")
    model = create_board_detector_model(input_size)
    model.summary()
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    # Split data for validation
    split_idx = int(len(X_train) * 0.8)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(X_train_split, y_train_split, batch_size=batch_size, subset='training'),
        epochs=epochs,
        validation_data=datagen.flow(X_train_split, y_train_split, batch_size=batch_size, subset='validation'),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # Print training summary
    print("\nTraining completed!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
    print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train board detector model')
    parser.add_argument('--data_dir', type=str, default='training_data/board_detector',
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='models/board_detector.h5',
                        help='Path to save trained model')
    parser.add_argument('--input_width', type=int, default=640,
                        help='Input image width')
    parser.add_argument('--input_height', type=int, default=480,
                        help='Input image height')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        input_size=(args.input_width, args.input_height),
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_path
    )
