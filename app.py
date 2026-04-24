from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Model aur Encoders load karein
try:
    with open("perfect_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        model = model_data['model']
        encoders = model_data['encoders']
        feature_names = model_data['feature_names']
    print("✅ Model and Encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 2. Input data ko encode karein (Exactly as per training)
        # Agar koi value dropdown se aisi aayi jo encoder mein nahi hai, toh handle karein
        city_enc = encoders['City'].transform([data['City']])[0]
        area_enc = encoders['Area'].transform([data['Area']])[0]
        prop_enc = encoders['Property_Type'].transform([data['Property_Type']])[0]
        facing_enc = encoders['Facing'].transform([data['Facing']])[0]

        # 3. Highway distances ka loop (Comparison ke liye)
        distances = [100, 500, 1000, 2000]
        predicted_results = {}

        for dist in distances:
            # Features ka wahi order hona chahiye jo training ke waqt tha
            # Order: City, Area, Property_Type, Size_sqft, Bedrooms, Bathrooms, Age_Years, Facing, Highway_Distance_m
            input_features = pd.DataFrame([[
                city_enc, 
                area_enc, 
                prop_enc, 
                float(data['Size_sqft']), 
                int(data['Bedrooms']), 
                int(data['Bathrooms']), 
                int(data['Age_Years']), 
                facing_enc, 
                float(dist)
            ]], columns=feature_names)

            prediction = model.predict(input_features)[0]
            # Price ko negative hone se bachayein (Just in case)
            predicted_results[str(dist)] = max(0, int(prediction))

        return jsonify({"predicted_prices": predicted_results})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Port 5000 par run karein
    app.run(host='0.0.0.0', port=5000, debug=True)