import json
import base64
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
import googlemaps
import folium
from streamlit_folium import st_folium
import requests

# Load API keys
GEMINI_API_KEY = st.secrets["APIKEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
GOOGLE_MAPS_KEY = st.secrets["GOOGLE_MAPS_KEY"]

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)

# Load model and class names
model = load_model('Mymodel.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Page config
st.set_page_config(page_title="🌿 AgriCure", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌱 AgriCure</h1>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

# Ask for location once image is uploaded
if uploaded_file:
    st.markdown("### 📷 Uploaded Image Preview", unsafe_allow_html=True)
    
    # Preview image centered
    encoded_img = base64.b64encode(uploaded_file.read()).decode()
    st.markdown(
        f"<div style='text-align:center'><img src='data:image/jpeg;base64,{encoded_img}' style='max-width:350px;border-radius:10px;'></div>",
        unsafe_allow_html=True
    )

    # Ask for location input
    user_location = st.text_input("📍 Enter your city, area, or pincode for local agro store suggestions:", "Kolkata")

    if user_location:
        with st.spinner("🤖 Predicting disease and fetching recommendations..."):

            # Step 1: Predict Disease
            uploaded_file.seek(0)  # Reset file pointer
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names[predicted_index]
            clean_name = predicted_class.replace("_", " ").title()
            confidence = np.max(prediction) * 100

            # Step 2: Generate Disease Info from Gemini
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            You are an agriculture expert. Explain the plant disease '{clean_name}' in simple terms.
            Include causes, symptoms, prevention, and commonly used treatments in India.
            """
            response = gemini_model.generate_content(prompt)
            gemini_text = response.text or "❌ No info found."

            # Step 3: Get location coordinates
            geocode_result = gmaps.geocode(user_location)
            if not geocode_result:
                map_html = "<p style='color:red'>❌ Could not find the entered location.</p>"
                map_display = False
            else:
                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lon = geocode_result[0]["geometry"]["location"]["lng"]

                # Step 4: Get nearby agro shops
                places = gmaps.places_nearby(
                    location=(lat, lon),
                    radius=5000,
                    keyword="agro shop"
                )

                # Create map
                map_obj = folium.Map(location=[lat, lon], zoom_start=13)
                folium.Marker(
                    [lat, lon], popup="Your Location", icon=folium.Icon(color="blue")
                ).add_to(map_obj)

                for place in places["results"]:
                    name = place["name"]
                    address = place.get("vicinity", "Address not available")
                    plat = place["geometry"]["location"]["lat"]
                    plon = place["geometry"]["location"]["lng"]
                    folium.Marker(
                        [plat, plon],
                        popup=f"{name}\n{address}",
                        icon=folium.Icon(color="green", icon="leaf")
                    ).add_to(map_obj)
                map_display = True

        # Display Prediction + Gemini Info + Map in Two Columns
        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center'><h2>✅ Predicted Disease: {clean_name}</h2><p>🧪 Model Confidence: {confidence:.2f}%</p></div>",
            unsafe_allow_html=True
        )
        
        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.subheader("📖 Disease Info & Prevention")
            st.markdown(gemini_text)
        with right_col:
            st.subheader("🗺️ Nearby Agro Stores")
            if map_display:
                st_folium(map_obj, width=350, height=500)
            else:
                st.warning("No map available for the entered location.")
