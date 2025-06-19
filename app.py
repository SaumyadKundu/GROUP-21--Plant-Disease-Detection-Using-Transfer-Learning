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

# --- API Keys ---
GEMINI_API_KEY = st.secrets["APIKEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
GOOGLE_MAPS_KEY = st.secrets["GOOGLE_MAPS_KEY"]

# --- Configuring APIs ---
genai.configure(api_key=GEMINI_API_KEY)
gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)

# --- Loading model and classes ---
model = load_model('Mymodel.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🌿 AgriCure", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌱 AgriCure</h1>", unsafe_allow_html=True)

# --- Uploading image ---
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown("### 📷 Uploaded Image Preview", unsafe_allow_html=True)

    # Centering image preview
    encoded_img = base64.b64encode(uploaded_file.read()).decode()
    st.markdown(
        f"<div style='text-align:center'><img src='data:image/jpeg;base64,{encoded_img}' style='max-width:350px;border-radius:10px;'></div>",
        unsafe_allow_html=True
    )

    # --- User location input ---
    user_location = st.text_input("📍 Enter your city, area, or pincode to find nearby agro stores:", "Kolkata")

    if user_location:
        with st.spinner("🤖 Predicting disease and fetching data..."):

            # --- Prediction ---
            uploaded_file.seek(0)
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names[predicted_index]
            clean_name = predicted_class.replace("_", " ").title()
            confidence = np.max(prediction) * 100

            # --- Gemini Description ---
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            You are an agriculture expert. Explain the plant disease '{clean_name}' in simple terms.
            Include causes, symptoms, prevention, and commonly used treatments in India.
            """
            response = gemini_model.generate_content(prompt)
            gemini_text = response.text or "❌ No info found."

            # --- Google Maps Nearby Shops ---
            map_display = False
            try:
                geocode_result = gmaps.geocode(user_location)
                if geocode_result:
                    lat = geocode_result[0]["geometry"]["location"]["lat"]
                    lon = geocode_result[0]["geometry"]["location"]["lng"]

                    places = gmaps.places_nearby(
                        location=(lat, lon),
                        radius=5000,
                        keyword="agro shop"
                    )

                    map_obj = folium.Map(location=[lat, lon], zoom_start=13)

                    # User location marker
                    folium.Marker(
                        [lat, lon],
                        popup="Your Location",
                        tooltip="📍 You are here",
                        icon=folium.Icon(color="blue")
                    ).add_to(map_obj)

                    for place in places["results"]:
                        name = place["name"]
                        address = place.get("vicinity", "Address not available")
                        plat = place["geometry"]["location"]["lat"]
                        plon = place["geometry"]["location"]["lng"]

                        folium.Marker(
                            [plat, plon],
                            popup=folium.Popup(f"<b>{name}</b><br>{address}", max_width=250),
                            tooltip=folium.Tooltip(name, sticky=True),
                            icon=folium.Icon(color="green", icon="leaf")
                        ).add_to(map_obj)

                    map_display = True
            except:
                map_display = False

            # --- SerpAPI Product Links ---
            serp_links = []
            try:
                query = f"{clean_name} plant disease medicine buy online"
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "gl": "in",
                    "hl": "en",
                    "num": "10"
                }
                serp_response = requests.get("https://serpapi.com/search", params=params)
                serp_results = serp_response.json()
                serp_links = serp_results.get("organic_results", [])
            except:
                serp_links = []

        # --- Displaying Output ---
        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center'><h2>✅ Predicted Disease: {clean_name}</h2></div>",
            unsafe_allow_html=True
        )

        # --- Two-Column Layout ---
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📖 Disease Info & Prevention")
            st.markdown(gemini_text)

            

        with right_col:
            with st.expander("🛒 Purchase Treatments Online", expanded=True):
                if serp_links:
                    for item in serp_links:
                        title = item.get("title", "")
                        link = item.get("link", "")
                        st.markdown(f"🔗 **[{title}]({link})**")
                else:
                    st.warning("No product results found online.")

            
            
            st.subheader("🗺️ Nearby Agro Stores")
            if map_display:
                # Lower height to reduce space
                _ = st_folium(map_obj, width=330, height=500, returned_objects=[])
            else:
                st.warning("⚠️ Map unavailable for the given location.")
        
            # Optional: slight spacing between map and links
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
         
            
