import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ----------------------------
# Helper function for prediction
# ----------------------------
def model_prediction(test_image):
    # Absolute path for model
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "trained_plant_disease_model.keras")
    
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch

    # Make prediction
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("FarmAssistX")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ----------------------------
# Load and display main image
# ----------------------------
base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, "Diseases.png")

if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img)
else:
    st.warning("Diseases.png not found in app directory.")

# ----------------------------
# Main Page
# ----------------------------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# ----------------------------
# Prediction Page
# ----------------------------
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            # Make prediction
            result_index = model_prediction(test_image)

            # Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            st.success(f"Model predicts: {class_name[result_index]}")
