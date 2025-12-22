from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib

# --- CONFIGURATION ---
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend"))
file_path = os.path.join(base_dir, 'shopping_behavior_updated.csv')
model_path = os.path.join(base_dir, 'model.pkl')

# Mock utils if missing
try:
    from utils import apply_feature_engineering
except ImportError:
    def apply_feature_engineering(df): return df

app = Flask(__name__, template_folder=frontend_dir, static_folder='static')

# --- LOAD MODEL ---
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("--> Model loaded successfully!")
    except: 
        print("--> Error loading model.")
else:
    print("--> WARNING: model.pkl not found.")

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    """Return all data for charts"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return jsonify(df.to_dict(orient='records'))
        return jsonify([])
    except Exception as e:
        print(f"Error: {e}")
        return jsonify([])

@app.route('/get_sample_data')
def get_sample_data():
    """Return 4000 random rows"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            sample_size = 4000 if len(df) >= 4000 else len(df)
            sample_df = df.sample(n=sample_size)
            return jsonify(sample_df.to_dict(orient='records'))
        else:
            return jsonify({'error': 'CSV file not found'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        # Mock response for testing frontend without model
        return jsonify({
            'prediction': "Yes (Demo)", 
            'prob_yes': 85.5, 
            'prob_no': 14.5,
            'message': "Model not found, running in Demo mode."
        })

    try:
        data = request.json
        raw_age = int(data['Age'])
        raw_rating = float(data['ReviewRating'])
        raw_amount = float(data['PurchaseAmount'])
        
        # Clamping
        safe_age = max(18, min(raw_age, 70))
        safe_rating = max(2.5, min(raw_rating, 5.0)) 
        safe_amount = max(20, min(raw_amount, 100))

        # Create Input DataFrame
        input_df = pd.DataFrame([{
            'Age': safe_age,
            'Gender': data['Gender'],
            'Category': data['Category'],
            'Purchase Amount (USD)': safe_amount,
            'Review Rating': safe_rating,
            'Previous Purchases': int(data['PreviousPurchases']),
            # Defaults
            'Season': 'Winter', 
            'Shipping Type': 'Express',
            'Payment Method': 'PayPal', 
            'Frequency of Purchases': 'Weekly'
        }])

        input_engineered = apply_feature_engineering(input_df)
        
        # Check features
        if hasattr(model, "n_features_in_") and input_engineered.shape[1] != model.n_features_in_:
             return jsonify({'error': 'Input features do not match model dimensions'}), 400

        # Predict
        prob_yes = model.predict_proba(input_engineered)[0][1]
        prob_no = 1 - prob_yes
        prediction_label = "Yes" if prob_yes >= 0.6 else "No"

        return jsonify({
            'prediction': prediction_label,
            'prob_yes': round(prob_yes * 100, 1),
            'prob_no': round(prob_no * 100, 1),
            'message': "Prediction successful"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)