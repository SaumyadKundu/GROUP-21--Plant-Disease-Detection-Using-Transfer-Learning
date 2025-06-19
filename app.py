import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
import requests

# Access API keys from Streamlit secrets
GEMINI_API_KEY = st.secrets["APIKEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model and class names
model = load_model('Mymodel.h5')

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Streamlit UI
st.set_page_config(page_title="PhytoScan", layout="centered")
st.title('🌿 PhytoScan - Plant Disease Identifier & Assistant')

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True, caption="Uploaded Image")

    with st.spinner("🧠 Predicting disease..."):
        # Preprocess image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_index]

        st.success(f"✅ Predicted Disease: **{predicted_class}**")

    # Gemini response
    with st.spinner("📚 Fetching description and prevention using AI..."):
        prompt = f"""You are an expert in agriculture. Write a detailed but simple description of the disease '{predicted_class}' in plants. 
        Include causes, symptoms, and prevention methods. Mention medicines or treatments available in India if possible."""
        
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        st.subheader("📖 Description and Prevention")
        st.markdown(response.text if response.text else "No description found.")

    # SerpAPI result
    with st.spinner("🔎 Searching for real products online..."):
        query = f"{predicted_class} fungicide"

        params = {
        "engine": "google",
        "q": query,
        "api_key": st.secrets["SERPAPI_KEY"],
        "gl": "in",
        "hl": "en",
        "num": "10"
}

        search_res = requests.get("https://serpapi.com/search", params=params)
        results = search_res.json()

        st.subheader("🛒 Available Products (Online Stores)")
        if "organic_results" in results:
            for item in results["organic_results"]:
                st.markdown(f"🔗 **{item['title']}**  \n[{item['link']}]({item['link']})")
        else:
            st.warning("No product results found.")
