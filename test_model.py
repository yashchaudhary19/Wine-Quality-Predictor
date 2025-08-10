import pickle
import numpy as np

def test_model():
    try:
        # Test loading the model
        print("Testing model loading...")
        with open('RF_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        print("✅ Model loaded successfully")
        
        # Test loading the scaler
        print("Testing scaler loading...")
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        print("✅ Scaler loaded successfully")
        
        # Test prediction
        print("Testing prediction...")
        test_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
        scaled_data = scaler.transform(test_data)
        prediction = model.predict(scaled_data)[0]
        print(f"✅ Prediction successful: {prediction}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_model()
