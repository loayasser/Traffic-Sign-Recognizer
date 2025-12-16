import numpy as np
import cv2
import os
import random
from tensorflow.keras.models import load_model

# 1. Load your trained model
# Make sure traffic_classifier.h5 is in the same folder!
print("Loading model...")
model = load_model('traffic_classifier.h5')
print("Model loaded successfully!")

# 2. Dictionary to label all traffic signs class.
classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

# 3. Pick a random image from the dataset to test
# (We use the data path again just to grab a file)
base_path = 'Train' # Make sure this matches your folder name!
random_class = random.randint(0, 42)
class_path = os.path.join(base_path, str(random_class))
images = os.listdir(class_path)
random_img = random.choice(images)
img_path = os.path.join(class_path, random_img)

# 4. Load and Preprocess the image (Must match training exactly!)
image = cv2.imread(img_path)
original_image = image.copy() # Keep a copy to display later
image = cv2.resize(image, (30, 30)) # Resize to 30x30
image = np.expand_dims(image, axis=0) # Add batch dimension: (30,30,3) -> (1,30,30,3)
image = image / 255.0 # Normalize 0-1

# 5. Predict
pred = model.predict(image)
class_index = np.argmax(pred) # Get the index of the highest probability
class_name = classes[class_index]

print(f"\nPREDICTION: {class_name}")
print(f"ACTUAL CLASS ID: {random_class}")

# 6. Show result
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title(f"AI Says: {class_name}")
plt.axis('off')
plt.show()