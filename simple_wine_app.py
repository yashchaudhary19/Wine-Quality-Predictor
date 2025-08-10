import streamlit as st
import pandas as pd
import pickle
import numpy as np
import streamlit as st
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded}");
             background-size: cover;
             background-position: top right;
             background-repeat: no-repeat;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Put this near the top of your main()
add_bg_from_local(r'C:\Users\chaud\OneDrive\Desktop\model1\wine-stains-vintage-linen-watercolour-backgrou-art-background-paint-watercolor-gradient_1020697-705638.avif')


# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('RF_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure RF_model.pkl and scaler.pkl are in the current directory.")
        return None, None

def main():
    st.set_page_config(
        page_title="Wine Quality Predictor",
        page_icon="🍷",
        layout="wide"
    )

    # Simple wine-themed styling
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(90deg, #8b0000, #b22222);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #8b0000;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-header">
            <h1>🍷 Wine Quality Prediction System</h1>
            <p>Predict wine quality based on physicochemical properties</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wine Properties")
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 15.9, 7.4, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.58, 0.52, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.27, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5, 0.1)
        chlorides = st.slider("Chlorides", 0.01, 0.61, 0.08, 0.001)
    
    with col2:
        st.subheader("More Properties")
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 15.0, 1.0)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 46.0, 1.0)
        density = st.slider("Density", 0.99, 1.004, 0.996, 0.0001)
        ph = st.slider("pH", 2.7, 4.0, 3.3, 0.01)
        sulphates = st.slider("Sulphates", 0.33, 2.0, 0.66, 0.01)
        alcohol = st.slider("Alcohol", 8.4, 14.9, 10.4, 0.1)
    
    # Prediction button
    if st.button("🔮 Predict Wine Quality", use_container_width=True):
        # Prepare input data
        input_data = np.array([
            [
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, ph, sulphates, alcohol
            ]
        ])
        
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Display result
            if prediction >= 7:
                quality_text = "Excellent 🍾"
                color = "#228B22"
                st.balloons()
            elif prediction >= 6:
                quality_text = "Good 👍"
                color = "#1E90FF"
            elif prediction >= 5:
                quality_text = "Average 😐"
                color = "#FFA500"
            else:
                quality_text = "Below Average 😬"
                color = "#B22222"
                st.snow()

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h2 style="color: {color}; text-align: center;">
                        Quality Score: {prediction}
                    </h2>
                    <h3 style="color: {color}; text-align: center;">
                        {quality_text}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show confidence
            st.subheader("Confidence Distribution")
            classes = model.classes_
            for i, (cls, prob) in enumerate(zip(classes, prediction_proba)):
                st.write(f"Quality {cls}: {prob:.2%}")
                st.progress(prob)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Information:**
        - Algorithm: Random Forest Classifier
        - Features: 11 physicochemical properties of wine
        - Target: Wine quality (scale 3-8)
        
        **Wine Quality Scale:**
        - 3-4: Poor quality
        - 5: Average quality  
        - 6: Good quality
        - 7-8: Excellent quality
        """)

if __name__ == "__main__":
    main()
