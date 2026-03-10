# 🚦 German Traffic Sign AI Recognition

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An end-to-end Deep Learning and Computer Vision application that identifies and classifies 43 distinct classes of German traffic signs. The model is built using a custom Convolutional Neural Network (CNN) and deployed as an interactive web application using Streamlit.

---

## 🚀 Key Features
* **High Accuracy:** Trained on 50,000+ real-world images handling various lighting, weather, and camera angles.
* **Custom Architecture:** Built a CNN from scratch using TensorFlow/Keras, utilizing Dropout layers to prevent overfitting.
* **Real-Time Inference:** Users can upload an image of a traffic sign to the Streamlit web app and get instant predictions.
* **Image Preprocessing:** Utilizes OpenCV for resizing, grayscaling (if applicable), and normalizing image arrays before feeding them to the model.

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow & Keras
* **Computer Vision:** OpenCV (`cv2`), Pillow (PIL)
* **Data Manipulation:** NumPy, Pandas
* **Deployment UI:** Streamlit

## 📊 The Dataset
The model was trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. It contains over 50,000 images categorized into 43 classes (e.g., Speed limits, Yield, Stop, No Entry, etc.). 

## 🧠 Model Architecture
The deep learning model utilizes a Sequential CNN architecture:
1. **Convolutional Layers (Conv2D):** To extract spatial features from the images.
2. **Pooling Layers (MaxPooling2D):** To downsample the feature maps and reduce computational load.
3. **Dropout Layers:** Applied strategically to prevent the model from overfitting to the training data.
4. **Fully Connected Layers (Dense):** To classify the flattened features into one of the 43 output classes using the Softmax activation function.

## 💻 How to Run Locally

Follow these steps to set up and run the project on your local machine:

**1. Clone the repository**
```bash
git clone [https://github.com/loayasser/Traffic-Sign-Recognizer.git](https://github.com/loayasser/Traffic-Sign-Recognizer.git)
cd Traffic-Sign-Recognizer