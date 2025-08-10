import streamlit as st
import pandas as pd
import pickle
import numpy as np
import random
import base64

# Helper to encode local image as base64 for CSS background
import os

def get_base64_bg(file):
    if not os.path.exists(file):
        st.warning(f"Image file not found: {file}")
        return ""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_bg("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Fpremium-ai-image%2Fwine-stains-vintage-linen-watercolour-backgrou-art-background-paint-watercolor-gradient_314973628.htm&psig=AOvVaw29V56sfs6swF-x2Hp_NGSK&ust=1754762563501000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCLCs34vm-44DFQAAAAAdAAAAABAV")


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

    # Enhanced wine-themed CSS with animations
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@300;400;600;700&display=swap');
        
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
            overflow-x: hidden;
        }}
        
        /* Wine bottle floating animation */
        .wine-bottle {{
            position: fixed;
            width: 60px;
            height: 120px;
            background: linear-gradient(45deg, #8b0000, #b22222);
            border-radius: 8px 8px 20px 20px;
            box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
            animation: float 6s ease-in-out infinite;
            z-index: 1;
        }}
        
        .wine-bottle::before {{
            content: '';
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 30px;
            background: linear-gradient(45deg, #8b0000, #b22222);
            border-radius: 10px 10px 0 0;
        }}
        
        .wine-bottle::after {{
            content: '🍷';
            position: absolute;
            top: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 20px;
        }}
        
        .wine-bottle:nth-child(1) {{
            left: 10%;
            animation-delay: 0s;
        }}
        
        .wine-bottle:nth-child(2) {{
            left: 85%;
            animation-delay: 2s;
        }}
        
        .wine-bottle:nth-child(3) {{
            left: 20%;
            top: 60%;
            animation-delay: 4s;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-20px) rotate(5deg); }}
        }}
        
        /* Grape cluster animation */
        .grape-cluster {{
            position: fixed;
            font-size: 24px;
            animation: grapeFloat 8s ease-in-out infinite;
            z-index: 1;
        }}
        
        .grape-cluster:nth-child(4) {{
            left: 75%;
            top: 20%;
            animation-delay: 1s;
        }}
        
        .grape-cluster:nth-child(5) {{
            left: 15%;
            top: 70%;
            animation-delay: 3s;
        }}
        
        .grape-cluster:nth-child(6) {{
            left: 80%;
            top: 80%;
            animation-delay: 5s;
        }}
        
        @keyframes grapeFloat {{
            0%, 100% {{ transform: translateY(0px) scale(1); }}
            50% {{ transform: translateY(-15px) scale(1.1); }}
        }}
        
        /* Sparkle animation */
        .sparkle {{
            position: fixed;
            width: 4px;
            height: 4px;
            background: #fff;
            border-radius: 50%;
            animation: sparkle 3s linear infinite;
            z-index: 1;
        }}
        
        .sparkle:nth-child(7) {{ left: 25%; top: 30%; animation-delay: 0s; }}
        .sparkle:nth-child(8) {{ left: 70%; top: 40%; animation-delay: 1s; }}
        .sparkle:nth-child(9) {{ left: 40%; top: 70%; animation-delay: 2s; }}
        .sparkle:nth-child(10) {{ left: 90%; top: 60%; animation-delay: 0.5s; }}
        .sparkle:nth-child(11) {{ left: 60%; top: 20%; animation-delay: 1.5s; }}
        
        @keyframes sparkle {{
            0%, 100% {{ opacity: 0; transform: scale(0); }}
            50% {{ opacity: 1; transform: scale(1); }}
        }}
        
        body {{
            background: transparent !important;
        }}
        
        html, body, [class*="css"]  {{
            font-family: 'Montserrat', sans-serif !important;
            color: #fff !important;
        }}
        
        .main-card {{
            background: rgba(30, 0, 30, 0.35);
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(44,0,22,0.25);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1.5px solid rgba(120,0,40,0.18);
            margin: 2rem auto 2rem auto;
            padding: 2.5rem 2.5rem 1.5rem 2.5rem;
            max-width: 900px;
            color: #fff !important;
            position: relative;
            z-index: 10;
            animation: cardGlow 4s ease-in-out infinite;
        }}
        
        @keyframes cardGlow {{
            0%, 100% {{ box-shadow: 0 8px 32px 0 rgba(44,0,22,0.25); }}
            50% {{ box-shadow: 0 8px 32px 0 rgba(139,0,0,0.3); }}
        }}
        
        h1, h2, h3, h4, h5, h6, .wine-title, .wine-subtitle {{
            color: #fff !important;
            text-shadow: 0 2px 8px #40001a99;
        }}
        
        label, .stSlider {{
            color: #fff !important;
        }}
        
        .stButton>button {{
            background: linear-gradient(90deg, #8b0000 0%, #b22222 100%);
            color: #fff;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1.15rem;
            padding: 0.6em 2.2em;
            border: none;
            box-shadow: 0 2px 12px 0 rgba(120,0,40,0.13);
            transition: 0.2s;
            animation: buttonPulse 2s ease-in-out infinite;
        }}
        
        @keyframes buttonPulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
        }}
        
        .stButton>button:hover {{
            background: linear-gradient(90deg, #b22222 0%, #8b0000 100%);
            color: #fff;
            transform: scale(1.05);
        }}
        
        .stProgress>div>div>div>div {{
            background-image: linear-gradient(90deg, #8b0000, #b22222);
        }}
        
        .st-cb {{
            border-radius: 14px !important;
        }}
        
        .wine-header {{
            text-align: center;
            margin-bottom: 0.5rem;
            animation: headerGlow 3s ease-in-out infinite;
        }}
        
        @keyframes headerGlow {{
            0%, 100% {{ text-shadow: 0 2px 8px #40001a99; }}
            50% {{ text-shadow: 0 2px 15px #8b0000; }}
        }}
        
        .wine-logo {{
            width: 70px;
            margin-bottom: 0.5rem;
            animation: logoSpin 10s linear infinite;
        }}
        
        @keyframes logoSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .wine-title {{
            font-family: 'Playfair Display', serif;
            font-size: 2.7rem;
            color: #8b0000;
            margin-bottom: 0.2rem;
            font-weight: 700;
        }}
        
        .wine-subtitle {{
            font-size: 1.2rem;
            color: #4b2e2b;
            margin-bottom: 1.2rem;
            font-weight: 600;
        }}
        
        .wine-footer {{
            text-align: center;
            color: #fff;
            font-size: 1.05rem;
            margin-top: 2.5rem;
            margin-bottom: 0.5rem;
            opacity: 0.85;
            letter-spacing: 0.5px;
        }}
        
        /* Remove default Streamlit backgrounds */
        .stApp, .block-container, .main, .css-18e3th9, .css-1d391kg {{
            background-color: rgba(0,0,0,0) !important;
        }}
        
        /* Enhanced slider styling */
        .stSlider > div > div > div > div {{
            background: linear-gradient(90deg, #8b0000, #b22222) !important;
        }}
        
        /* Prediction result animation */
        .prediction-result {{
            animation: resultGlow 2s ease-in-out infinite;
        }}
        
        @keyframes resultGlow {{
            0%, 100% {{ box-shadow: 0 2px 12px 0 rgba(139,0,0,0.2); }}
            50% {{ box-shadow: 0 2px 20px 0 rgba(139,0,0,0.4); }}
        }}
        </style>
        
        <!-- Animated wine elements -->
        <div class="wine-bottle"></div>
        <div class="wine-bottle"></div>
        <div class="wine-bottle"></div>
        <div class="grape-cluster">🍇</div>
        <div class="grape-cluster">🍇</div>
        <div class="grape-cluster">🍇</div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        <div class="sparkle"></div>
        """,
        unsafe_allow_html=True
    )

    # --- Main frosted glass card ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    # --- Custom Header with Logo and Subtitle ---
    st.markdown(
        """
        <div class="wine-header">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="wine-logo" alt="Wine Logo" />
            <div class="wine-title">Wine Quality Prediction System</div>
            <div class="wine-subtitle">Discover the art and science of wine, one prediction at a time 🍷</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Centered main title
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0.5em;'>🍷 Wine Quality Prediction System</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>Enter Wine Characteristics to Predict Quality</h3>
        <p>This model predicts wine quality on a scale from 3-8 based on physicochemical properties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Acidity & pH Properties")
        fixed_acidity = st.slider(
            "Fixed Acidity", 
            min_value=3.8, max_value=15.9, value=7.4, step=0.1,
            help="Non-volatile acids that don't evaporate readily"
        )
        
        volatile_acidity = st.slider(
            "Volatile Acidity", 
            min_value=0.08, max_value=1.58, value=0.52, step=0.01,
            help="Amount of acetic acid in wine (high levels lead to unpleasant vinegar taste)"
        )
        
        citric_acid = st.slider(
            "Citric Acid", 
            min_value=0.0, max_value=1.0, value=0.27, step=0.01,
            help="Adds freshness and flavor to wines"
        )
        
        ph = st.slider(
            "pH", 
            min_value=2.7, max_value=4.0, value=3.3, step=0.01,
            help="Describes how acidic or basic a wine is (0-14 scale)"
        )
        
        st.subheader("Sulfur Content")
        free_sulfur_dioxide = st.slider(
            "Free Sulfur Dioxide", 
            min_value=1.0, max_value=72.0, value=15.0, step=1.0,
            help="Prevents microbial growth and oxidation of wine"
        )
        
        total_sulfur_dioxide = st.slider(
            "Total Sulfur Dioxide", 
            min_value=6.0, max_value=289.0, value=46.0, step=1.0,
            help="Amount of free and bound forms of SO2"
        )
    
    with col2:
        st.subheader("Chemical Properties")
        residual_sugar = st.slider(
            "Residual Sugar", 
            min_value=0.9, max_value=15.5, value=2.5, step=0.1,
            help="Amount of sugar remaining after fermentation"
        )
        
        chlorides = st.slider(
            "Chlorides", 
            min_value=0.01, max_value=0.61, value=0.08, step=0.001,
            help="Amount of salt in the wine"
        )
        
        density = st.slider(
            "Density", 
            min_value=0.99, max_value=1.004, value=0.996, step=0.0001,
            help="Density of wine (depends on alcohol and sugar content)"
        )
        
        sulphates = st.slider(
            "Sulphates", 
            min_value=0.33, max_value=2.0, value=0.66, step=0.01,
            help="Wine additive that contributes to SO2 levels"
        )
        
        alcohol = st.slider(
            "Alcohol", 
            min_value=8.4, max_value=14.9, value=10.4, step=0.1,
            help="Alcohol percentage by volume"
        )
    
    st.markdown("---")
    
    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Wine Quality", use_container_width=True):
            # Prepare input data
            input_data = np.array([
                [
                    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, ph, sulphates, alcohol
                ]
            ])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # --- Animated Results ---
            if prediction >= 7:
                color = "#228B22"
                quality_text = "Excellent 🍾"
                st.balloons()
                icon = "🥇"
            elif prediction >= 6:
                color = "#1E90FF"
                quality_text = "Good 👍"
                icon = "🍷"
            elif prediction >= 5:
                color = "#FFA500"
                quality_text = "Average 😐"
                icon = "🍇"
            else:
                color = "#B22222"
                quality_text = "Below Average 😬"
                st.snow()
                icon = "🍂"

            st.markdown(f"""
            <div class="prediction-result" style='text-align: center; padding: 24px; border-radius: 16px; 
                        background: #fff8f5; border: 2px solid {color}; box-shadow: 0 2px 12px 0 {color}22;'>
                <h2 style='color: {color}; margin: 0;'>{icon} Quality Score: {prediction}</h2>
                <h3 style='color: {color}; margin: 10px 0;'>{quality_text}</h3>
            </div>
            """, unsafe_allow_html=True)

            # --- Progress Bars with Icons ---
            st.markdown("### 📊 Confidence Distribution")
            classes = model.classes_
            probabilities = prediction_proba
            prob_df = pd.DataFrame({
                'Quality Level': classes,
                'Confidence': probabilities
            }).sort_values('Quality Level')
            for _, row in prob_df.iterrows():
                bar_icon = "🍷" if row['Quality Level'] >= 7 else ("🍇" if row['Quality Level'] >= 5 else "🍂")
                st.write(f"{bar_icon} Quality {int(row['Quality Level'])}")
                st.progress(row['Confidence'])
                st.write(f"{row['Confidence']:.2%}")
    
    st.markdown("---")
    
    # Additional information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Information:**
        - Algorithm: Random Forest Classifier
        - Features: 11 physicochemical properties of wine
        - Target: Wine quality (scale 3-8)
        - Data preprocessing: SMOTE oversampling and Standard Scaling
        
        **Wine Quality Scale:**
        - 3-4: Poor quality
        - 5: Average quality  
        - 6: Good quality
        - 7-8: Excellent quality
        
        **Features Used:**
        1. **Fixed Acidity**: Tartaric acid - contributes to wine's taste
        2. **Volatile Acidity**: Acetic acid - high levels can lead to unpleasant taste
        3. **Citric Acid**: Preservative that adds freshness
        4. **Residual Sugar**: Sugar left after fermentation
        5. **Chlorides**: Salt content
        6. **Free Sulfur Dioxide**: Prevents microbial growth
        7. **Total Sulfur Dioxide**: Total SO2 (free + bound)
        8. **Density**: Depends on alcohol and sugar content
        9. **pH**: Acidity level (lower = more acidic)
        10. **Sulphates**: Wine additive (potassium sulphate)
        11. **Alcohol**: Percentage by volume
        """)
    
    # Sample data for testing
    with st.expander("🧪 Try Sample Wines"):
        st.markdown("Click on these buttons to test with sample wine data:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("High Quality Wine"):
                st.session_state.update({
                    'fixed_acidity': 7.4, 'volatile_acidity': 0.7, 'citric_acid': 0.0,
                    'residual_sugar': 1.9, 'chlorides': 0.076, 'free_sulfur_dioxide': 11.0,
                    'total_sulfur_dioxide': 34.0, 'density': 0.9978, 'ph': 3.51,
                    'sulphates': 0.56, 'alcohol': 9.4
                })
        
        with col2:
            if st.button("Average Quality Wine"):
                st.session_state.update({
                    'fixed_acidity': 8.1, 'volatile_acidity': 0.87, 'citric_acid': 0.0,
                    'residual_sugar': 4.1, 'chlorides': 0.095, 'free_sulfur_dioxide': 5.0,
                    'total_sulfur_dioxide': 14.0, 'density': 0.99854, 'ph': 3.36,
                    'sulphates': 0.53, 'alcohol': 9.6
                })
        
        with col3:
            if st.button("Low Quality Wine"):
                st.session_state.update({
                    'fixed_acidity': 11.2, 'volatile_acidity': 0.28, 'citric_acid': 0.56,
                    'residual_sugar': 1.9, 'chlorides': 0.075, 'free_sulfur_dioxide': 17.0,
                    'total_sulfur_dioxide': 60.0, 'density': 0.998, 'ph': 3.16,
                    'sulphates': 0.58, 'alcohol': 9.8
                })

    # --- Footer ---
    st.markdown(
        """
        </div>
        <div class="wine-footer">
            Made with ❤️ by <b>Your Name</b> &middot; <a href="https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009" style="color:#fff;text-decoration:underline;" target="_blank">Wine Data</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
