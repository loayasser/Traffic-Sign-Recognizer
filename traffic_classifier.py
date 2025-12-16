import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# --- CONFIGURATION ---
# UPDATE THIS PATH if needed! 
# If your folder structure is Traffic -> Train -> 0,1,2... use 'Traffic/Train'
DATA_PATH = 'Train' 
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3
NUM_CATEGORIES = 43

# --- 1. VERIFY PATH ---
if not os.path.exists(DATA_PATH):
    print(f"ERROR: The folder '{DATA_PATH}' was not found.")
    print(f"Current working directory: {os.getcwd()}")
    exit()

# --- 2. LOAD DATA ---
data = []
labels = []

print("Start loading data... (This might take 1-2 minutes)")

for i in range(NUM_CATEGORIES):
    path = os.path.join(DATA_PATH, str(i))
    if not os.path.exists(path):
        continue
        
    images = os.listdir(path)
    
    # Simple progress indicator
    if i % 10 == 0:
        print(f"Loading Class {i}...")

    for img in images:
        try:
            image = cv2.imread(os.path.join(path, img))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image: {img}")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Data loaded successfully!")
print(f"Total Images: {data.shape[0]}")

# --- 3. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# --- 4. VISUALIZE (Before Normalization) ---
# We visualize here while data is still integers (0-255) to avoid errors
import random
idx = random.randint(0, len(X_train))

plt.figure(figsize=(5,5))
# We manually swap channels from BGR (OpenCV) to RGB (Matplotlib) without using cvtColor
# This prevents the specific error you were seeing
plt.imshow(X_train[idx][:, :, ::-1]) 
plt.title(f"Traffic Sign Class ID: {y_train[idx]}")
plt.axis('off')
plt.show() 
# CODE WILL PAUSE HERE UNTIL YOU CLOSE THE IMAGE WINDOW

# --- 5. PREPROCESS ---
print("Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train, NUM_CATEGORIES)
y_test_cat = to_categorical(y_test, NUM_CATEGORIES)

# --- 6. BUILD MODEL ---
print("\nBuilding the Neural Network...")
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)),
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.5),
    Dense(NUM_CATEGORIES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 7. TRAIN ---
print("\nStarting Training... (Grab a coffee, this takes time!)")
history = model.fit(X_train, y_train_cat, batch_size=64, epochs=15, validation_data=(X_test, y_test_cat))

print("Training Complete!")

# --- 8. PLOT RESULTS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

model.save('traffic_classifier.h5')
print("Model saved!")