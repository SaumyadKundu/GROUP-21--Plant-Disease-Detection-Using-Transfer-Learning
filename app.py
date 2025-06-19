import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
from serpapi.google_search import GoogleSearch

# Accessing API keys from Streamlit secrets
api_key = st.secrets["APIKEY"]
serpapi_key = st.secrets["SERPAPI_KEY"]

# Configure Gemini AI
genai.configure(api_key=api_key)

# Load model
model = load_model('Mymodel.h5')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Streamlit UI
st.title('🌿 PhytoScan - Plant Disease Identifier & Assistant')

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    st.markdown(f"### 🧬 Predicted Disease: **{predicted_class_name}**")

    # Gemini AI for description
    with st.spinner("Fetching disease info..."):
        prompt = f"Write a detailed description and prevention method for the disease {predicted_class_name} in plants. List medicines or treatments sold in India."
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)

        if response.text:
            st.subheader("🧾 Description and Prevention")
            st.write(response.text)

    # SerpAPI for real product links
    with st.spinner("Searching for real-world treatments..."):
        params = {
            "engine": "google",
            "q": f"{predicted_class_name} fungicide site:agribegri.com OR site:amazon.in OR site:flipkart.com",
            "location": "India",
            "hl": "en",
            "gl": "in",
            "api_key": serpapi_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results:
            st.subheader("🛒 Available Products Online")
            for res in results["organic_results"][:5]:  # show top 5 results
                st.markdown(f"**{res['title']}**\n\n[{res['link']}]({res['link']})")
        else:
            st.write("Couldn't fetch product links. Try again later.")
