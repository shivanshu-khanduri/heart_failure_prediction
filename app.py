
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('heart_failure_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"Model loaded successfully: {model_info['model_name']}")
    print(f"Model accuracy: {model_info['accuracy']:.4f}")
except FileNotFoundError:
    print("Model files not found. Please run the training script first.")
    model = None
    scaler = None
    model_info = None

# Feature names and their descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'anaemia': 'Anaemia (0: No, 1: Yes)',
    'creatinine_phosphokinase': 'Creatinine Phosphokinase (mcg/L)',
    'diabetes': 'Diabetes (0: No, 1: Yes)',
    'ejection_fraction': 'Ejection Fraction (%)',
    'high_blood_pressure': 'High Blood Pressure (0: No, 1: Yes)',
    'platelets': 'Platelets (kiloplatelets/mL)',
    'serum_creatinine': 'Serum Creatinine (mg/dL)',
    'serum_sodium': 'Serum Sodium (mEq/L)',
    'sex': 'Sex (0: Female, 1: Male)',
    'smoking': 'Smoking (0: No, 1: Yes)',
    'time': 'Follow-up Period (days)'
}

# Normal ranges for reference
NORMAL_RANGES = {
    'age': (0, 100),
    'anaemia': (0, 1),
    'creatinine_phosphokinase': (0, 10000),
    'diabetes': (0, 1),
    'ejection_fraction': (10, 80),
    'high_blood_pressure': (0, 1),
    'platelets': (50000, 500000),
    'serum_creatinine': (0.5, 5.0),
    'serum_sodium': (125, 150),
    'sex': (0, 1),
    'smoking': (0, 1),
    'time': (1, 300)
}

@app.route('/')
def home():
    if model is None:
        return "Model not loaded. Please place 'heart_failure_model.pkl', 'scaler.pkl', and 'model_info.json' in the same folder as app.py."
    return render_template('index.html', 
                         features=FEATURE_DESCRIPTIONS,
                         ranges=NORMAL_RANGES,
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        features = []
        feature_names = model_info['features']
        
        for feature in feature_names:
            value = float(request.form[feature])
            features.append(value)
        
        features_array = np.array([features])
        
        if model_info['use_scaling']:
            features_array = scaler.transform(features_array)
        
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        result = {
            'prediction': int(prediction),
            'probability_no_death': float(probability[0]),
            'probability_death': float(probability[1]),
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = []
        feature_names = model_info['features']
        
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(float(data[feature]))
        
        features_array = np.array([features])
        
        if model_info['use_scaling']:
            features_array = scaler.transform(features_array)
        
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        result = {
            'prediction': int(prediction),
            'probability_no_death': float(probability[0]),
            'probability_death': float(probability[1]),
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low',
            'model_info': {
                'model_name': model_info['model_name'],
                'accuracy': model_info['accuracy']
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_info': model_info if model_info else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
