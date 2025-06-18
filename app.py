import json
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup


# Accessing the API key from Streamlit secrets
api_key = st.secrets["APIKEY"]

# Configuring the API key for Gemini AI
genai.configure(api_key=api_key)

# Loading the trained model 
model = load_model('Mymodel.h5')


# Loading class names from JSON file
class_names_path = 'class_names.json'
with open(class_names_path, 'r') as f:
    class_names = json.load(f)
    
#####################################
def fetch_medicines_agribegri(disease_name, max_results=3):
    import requests
    from bs4 import BeautifulSoup

    query = f"{disease_name} fungicide site:agribegri.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={query}"

    res = requests.get(search_url, headers=headers)
    if res.status_code != 200:
        return [{"error": "Failed to fetch search results from Google"}]

    soup = BeautifulSoup(res.text, "html.parser")
    results = []

    for tag in soup.find_all("a"):
        href = tag.get("href", "")
        if "agribegri.com" in href and "/url?q=" in href:
            # Clean the Google redirect URL
            url = href.split("/url?q=")[1].split("&")[0]

            # Filter out non-product pages
            if any(skip in url.lower() for skip in ["login", "account", "terms"]):
                continue

            title = tag.text.strip()
            if not title or len(title) < 8:
                title = f"{disease_name} Treatment Product"

            results.append({
                "product": title,
                "link": url
            })

        if len(results) >= max_results:
            break

    return results if results else [{"error": "No relevant results found on AgriBegri"}]



#################################

# Streamlit app interface
st.title('PhytoScan')

# Uploading image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Displaying uploaded image
    st.image(uploaded_file, use_container_width=True)

    # Loading image and preparing it for prediction
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjusting the size according to our model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    img_array /= 255.0  # Normalizing the image

    # Predicting the class
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Geting the class with highest probability
    predicted_class_name = class_names[predicted_class_index]  # Maping index to class name

    # Showing the prediction result
    st.write(f"Predicted Disease: {predicted_class_name}")

    # Using Gemini AI to generate description and prevention content
    prompt = f"Write a detailed description and prevention method for the disease {predicted_class_name} in plants. List medicines or treatments sold in India for {predicted_class_name} and include known websites or marketplaces if possible"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
#############################
        # Scrape medicine links from IndiaMart
    st.subheader("Medicines from AgriBegri (India)")
medicines = fetch_medicines_agribegri(predicted_class_name)

for med in medicines:
    if 'error' in med:
        st.warning(med['error'])
    else:
        st.markdown(f"🔹 **{med['product']}**")
        st.markdown(f"[🛒 Buy Now]({med['link']})")




#################################
    # Showing the generated description and prevention
    if response.text:
        st.subheader("Description and Prevention")
        st.write(response.text)
    else:
        st.write("No detailed information available for this disease.")
