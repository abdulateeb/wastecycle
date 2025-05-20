import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Waste Material Classifier", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return YOLO('runs/classify/waste_classifier/weights/best.pt')

def main():
    st.title("Waste Material Classification")
    st.write("Upload an image to classify the type of waste material")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        model = load_model()
        results = model.predict(image)[0]
        
        with col2:
            st.subheader("Prediction Results:")
            # Get the predicted class and confidence
            predicted_class = results.names[results.probs.top1]
            confidence = float(results.probs.top1conf)
            
            # Display results with custom styling
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
                <h3>Predicted Waste Type:</h3>
                <h2 style='color: #1f77b4;'>{predicted_class.replace('_', ' ').title()}</h2>
                <h3>Confidence Score:</h3>
                <h2 style='color: #2ecc71;'>{confidence:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
