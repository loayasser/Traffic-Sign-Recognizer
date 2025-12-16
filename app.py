import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# --- PART 1: LOAD THE BRAIN (THE MODEL) ---
# We use @st.cache so the app doesn't have to reload the model every time you click a button.
# It loads once and stays ready.
@st.cache_resource
def load_my_model():
    # This expects the model file to be in the same folder
    model = tf.keras.models.load_model('traffic_classifier.h5')
    return model

model = load_my_model()

# --- PART 2: DEFINE THE DICTIONARY (THE TRANSLATOR) ---
# The model only knows numbers (0, 1, 2...). We need this list to turn 
# "Class 0" into "Speed Limit 20km/h".
classes = { 
    0:'Speed limit (20km/h)', 
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

# --- PART 3: BUILD THE WEBSITE (THE UI) ---
st.title("🚦 Traffic Sign Recognizer AI")
st.write("Upload an image of a traffic sign, and I will tell you what it is!")

# Create the file uploader button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Show the user the image they uploaded
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 2. Add a button to trigger the prediction
    if st.button('Identify Sign'):
        st.write("Analyzing...")
        
        # --- PART 4: PREPARE THE IMAGE FOR THE CHEF ---
        # Convert the image to a numpy array
        img_array = np.array(image)
        
        # Resize it to 30x30 pixels (Because that's what we trained the model on!)
        # If we don't resize, the model will crash.
        img_array = cv2.resize(img_array, (30, 30))
        
        # Convert to batch format (1, 30, 30, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values to be between 0 and 1
        # (Only do this IF you divided by 255 during training. If you didn't, remove this line)
        # Assuming we did standard normalization:
        # img_array = img_array / 255.0  <-- UNCOMMENT IF YOUR ACCURACY IS BAD
        
        # --- PART 5: PREDICT ---
        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        confidence = np.max(pred) * 100
        
        result = classes[class_index]
        
        # Show the result in big green text
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: {confidence:.2f}%")