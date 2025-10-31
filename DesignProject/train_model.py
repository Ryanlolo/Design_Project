import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
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

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')  # empty, red, blue
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def load_training_data(data_dir):
    images = []
    labels = []
    
    classes = ['empty', 'red', 'blue']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                img = img / 255.0  # normalization
                
                images.append(img)
                
                # Create one-hot tag
                label = [0, 0, 0]
                label[class_idx] = 1
                labels.append(label)
    
    return np.array(images), np.array(labels)

def train():
    # Load data
    print("Load training data...")
    X_train, y_train = load_training_data('training_data')
    
    if len(X_train) == 0:
        print("No training data found, please collect data first")
        return
    
    print(f"Amount of training data: {len(X_train)}")
    
    # Create model
    model = create_model()
    
    # data enhancement
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Training model
    print("Start training the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32, subset='training'),
        epochs=30,
        validation_data=datagen.flow(X_train, y_train, batch_size=32, subset='validation'),
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/piece_classifier.h5')
    print("Model saved to models/piece_classifier.h5")

if __name__ == "__main__":
    train()