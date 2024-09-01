import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import io

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = Image.open(test_image).convert('RGB')  # Convert image to RGB
    image = image.resize((128, 128))  # Resize to match model's input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dims to match model's input shape
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Disease Recognition", "About Us"])

# Prediction Page
if app_mode == "Disease Recognition":
    st.header("About Plant Disease Predictor AI")
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            class_name = [
                "Apple scab",
                "Apple Black rot",
                "Apple Cedar apple rust",
                "Apple healthy",
                "Blueberry healthy",
                "Cherry (including sour) healthy",
                "Cherry (including sour) Powdery mildew",
                "Corn (maize) Cercospora_leaf spot Gray leaf spot",
                "Corn (maize) Common rust ",
                "Corn (maize) healthy",
                "Corn (maize) Northern Leaf Blight",
                "Grape Black rot",
                "Grape Esca (Black Measles)",
                "Grape healthy",
                "Grape Leaf blight (Isariopsis Leaf Spot)",
                "Orange Haunglongbing (Citrus greening)",
                "Peach Bacterial spot",
                "Peach healthy",
                "Pepper, bell Bacterial spot",
                "Pepper, bell healthy",
                "Potato Early blight",
                "Potato healthy",
                "Potato Late blight",
                "Raspberry healthy",
                "Soybean healthy",
                "Squash Powdery mildew",
                "Strawberry healthy",
                "Strawberry Leaf scorch",
                "Tomato Bacterial spot",
                "Tomato Early blight",
                "Tomato healthy",
                "Tomato Late blight",
                "Tomato Leaf Mold",
                "Tomato Septoria leaf spot",
                "Tomato Spider mites Two-spotted spider mite",
                "Tomato Target Spot",
                "Tomato mosaic virus",
                "Tomato Yellow Leaf Curl Virus"
            ]
            st.success(f"Model is predicting it's a : {class_name[result_index]}")

# About Page
elif app_mode == "About Us":
    st.header("Crop Disease Detector")
    st.image("about_page_background.jpg", use_column_width=True)
    st.markdown('''
**Plant Disease Predictor AI** helps you detect and manage plant diseases early. Our mission is to empower you with AI-driven tools for accurate and quick diagnosis, ensuring healthy crops and reducing losses.

**How It Works**:
Upload an Image: Submit a clear photo of your plant.
AI Analysis: Our AI identifies any disease in the image.
Instant Results: Get a quick diagnosis and treatment advice.

**Why Us?**
- **Accurate**: High precision in detecting plant diseases.
- **Easy to Use**: No technical skills needed.
- **Comprehensive Support**: Actionable insights for effective treatment.

**Our Technology**
We use cutting-edge AI and machine learning, continuously updated to improve accuracy and incorporate the latest research.

**Join Us**
Protect your plants with technology. Whether you're a farmer or a plant lover, our AI is here to help you maintain healthy crops.
    ''')
