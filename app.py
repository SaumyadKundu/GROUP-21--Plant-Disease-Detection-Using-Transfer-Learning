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

# Page configuration
st.set_page_config(page_title="🌿 AgriCure - Plant Disease Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌱 AgriCure</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect plant diseases using AI and find treatment solutions instantly.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("📤 Upload a clear image of the affected plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True, caption="📷 Uploaded Leaf Image")
    st.markdown("---")

    with st.spinner("🔍 Analyzing the image to predict disease..."):
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_index]

        st.success(f"✅ **Predicted Disease:** {predicted_class}")
        st.markdown("---")

    # Create two-column layout
    left_col, right_col = st.columns([2, 1])  # 2:1 width ratio

    with left_col:
        with st.spinner("📚 Fetching expert description and prevention steps..."):
            prompt = f"""
            You are an expert in agriculture. Write a clear, simple description of the disease '{predicted_class}' in plants. 
            Include causes, symptoms, prevention methods, and commonly used treatments or medicines in India.
            """
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)

            st.subheader("📖 Disease Information & Prevention Tips")
            st.markdown(response.text if response.text else "❌ No information found.")

    with right_col:
        with st.spinner("🛒 Looking up available treatments online..."):
            query = f"{predicted_class} medicine buy online"
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_KEY,
                "gl": "in",
                "hl": "en",
                "num": "10"
            }

            search_res = requests.get("https://serpapi.com/search", params=params)
            results = search_res.json()

            st.subheader("🛍️ Purchase Treatments Online")
            if "organic_results" in results:
                for item in results["organic_results"]:
                    st.markdown(f"🔗 **[{item['title']}]({item['link']})**")
            else:
                st.warning("⚠️ No relevant products found online.")

    st.markdown("---")

# Footer
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Made with ❤️ for Indian Farmers | AgriCure 2025</p>",
    unsafe_allow_html=True
)
