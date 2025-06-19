import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
import requests

# Accessing API keys from Streamlit secrets
GEMINI_API_KEY = st.secrets["APIKEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# Configuring Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Loading model and class names
model = load_model('Mymodel.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Page configuration
st.set_page_config(page_title="🌿 AgriCure - Plant Disease Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌱 AgriCure</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect plant diseases using AI and find treatment solutions instantly.</p>", unsafe_allow_html=True)
st.markdown("---")

# Uploading image
uploaded_file = st.file_uploader("📤 Upload a clear image of the affected plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, width=350, caption="📷 Uploaded Leaf Image")
    st.markdown("---")

    with st.spinner("🔍 Analyzing the image to predict disease..."):
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_index]

        clean_name = predicted_class.replace("_", " ").title()

        # Display nicely
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 20px;'>
                <span style='font-size: 28px;'>✅ <strong>Predicted Disease:</strong> {clean_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

    # Creating two-column layout
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
######################

import googlemaps
import folium
from streamlit_folium import st_folium

# --- Section Title ---
st.subheader("📍 Find Nearby Agro Shops or Krishi Centers")

# Input from user (city or pincode)
user_location = st.text_input("Enter your location (city, pincode, or area)", "Kolkata")

if st.button("🔍 Search Nearby Agro Shops"):
    with st.spinner("Fetching nearby stores using Google Places API..."):
        try:
            # Initialize Google Maps client
            gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_KEY"])

            # Get latitude and longitude from address
            geocode_result = gmaps.geocode(user_location)
            if not geocode_result:
                st.error("❌ Could not find location. Try a valid city or pin code.")
            else:
                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lon = geocode_result[0]["geometry"]["location"]["lng"]

                # Get nearby agro shops
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

                found = False
                for place in places["results"]:
                    name = place["name"]
                    address = place.get("vicinity", "No address")
                    lat_p = place["geometry"]["location"]["lat"]
                    lon_p = place["geometry"]["location"]["lng"]
                    folium.Marker(
                        [lat_p, lon_p],
                        popup=f"{name}\n{address}",
                        icon=folium.Icon(color="green", icon="leaf")
                    ).add_to(map_obj)
                    found = True

                if found:
                    st.success("✅ Found agro shops nearby.")
                    st_folium(map_obj, width=700, height=500)
                else:
                    st.warning("⚠️ No agro shops found nearby.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
#####################
# Footer
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: gray;'>Made with ❤️ for Indian Farmers | AgriCure 2025</p>",
    unsafe_allow_html=True
)
