import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai

# Access the API key from Streamlit secrets
api_key = st.secrets["APIKEY"]

# Configure the API key for Gemini AI
genai.configure(api_key=api_key)
# Load the trained model (replace with the correct path to your model)
# model = load_model(r'D:\1. LONEWALKER\1.1 CODES\FINAL YEAR PROJECTS REPO\1_Streamlit\Mymodel.h5')
model = load_model('Mymodel.h5')

# Path to your dataset directory (this should be where the 'train' or 'valid' folders are located)
# dataset_dir = r'D:\1. LONEWALKER\1.1 CODES\FINAL YEAR PROJECTS REPO\1_Streamlit\train'  # Update this path to your actual dataset directory

# Get class names from folder names in the dataset directory
# class_names = sorted(os.listdir(dataset_dir))  # comment- List and sort folder names to get class names
# Load class names from JSON file
# class_names_path = r'D:\1. LONEWALKER\1.1 CODES\FINAL YEAR PROJECTS REPO\1_Streamlit\class_names.json'
class_names_path = 'class_names.json'
with open(class_names_path, 'r') as f:
    class_names = json.load(f)


# Streamlit app interface
st.title('PhytoScan')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, use_container_width=True)

    # Load image and prepare it for prediction
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust the size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the class with highest probability
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name

    # Show the prediction result
    st.write(f"Predicted Disease: {predicted_class_name}")

    # Use Gemini AI to generate description and prevention content
    prompt = f"Write a detailed description and prevention method for the disease {predicted_class_name} in plants."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Show the generated description and prevention
    if response.text:
        st.subheader("Description and Prevention")
        st.write(response.text)
    else:
        st.write("No detailed information available for this disease.")
