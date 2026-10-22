import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained SVC model
with open('svc.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "SVC Model API is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Extract features in the order expected by the model
        features = [
            data.get('age'),
            data.get('gender'),
            data.get('city'),
            data.get('bmi'),
            data.get('family_history_diabetes'),
            data.get('physical_activity_level'),
            data.get('diet_type'),
            data.get('smoking_status'),
            data.get('alcohol_consumption'),
            data.get('hours_sleep_per_night'),
            data.get('stress_level'),
            data.get('fasting_blood_sugar'),
            data.get('hba1c_level'),
            data.get('blood_pressure_systolic'),
            data.get('blood_pressure_diastolic'),
            data.get('waist_circumference_cm'),
            data.get('income_bracket')
        ]
        
        # Note: If your model was trained on numerically encoded categorical variables 
        # (like gender, city, diet_type, etc.), make sure the incoming JSON values 
        # are already encoded, or apply your encoder here before predicting.
        
        prediction = model.predict([features])
        
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
