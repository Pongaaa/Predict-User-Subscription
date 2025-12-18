# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib # Dùng để load model
from utils import apply_feature_engineering # Import hàm chung

# Cấu hình đường dẫn
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend"))
file_path = os.path.join(base_dir, 'shopping_behavior_updated.csv')
model_path = os.path.join(base_dir, 'model.pkl')

# Kiểm tra frontend
if not os.path.exists(os.path.join(frontend_dir, 'index.html')):
    if os.path.exists(os.path.join(base_dir, 'index.html')):
        frontend_dir = base_dir

app = Flask(__name__, template_folder=frontend_dir, static_folder=frontend_dir)

# --- LOAD MODEL ĐÃ TRAIN ---
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("--> Đã load model thành công!")
else:
    model = None
    print("--> CẢNH BÁO: Chưa có file model.pkl. Hãy chạy 'python train.py' trước!")

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    try:
        df = pd.read_csv(file_path)
        return jsonify(df.head(50).to_dict(orient='records'))
    except Exception as e:
        return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model chưa được huấn luyện'}), 500

    data = request.json
    
    # 1. Lấy dữ liệu
    raw_age = int(data['Age'])
    raw_rating = float(data['ReviewRating'])
    raw_amount = float(data['PurchaseAmount'])
    
    # 2. Clamping
    safe_age = max(18, min(raw_age, 70))
    safe_rating = max(2.5, min(raw_rating, 5.0)) 
    safe_amount = max(20, min(raw_amount, 100))

    # 3. Tạo DataFrame input
    input_df = pd.DataFrame([{
        'Age': safe_age,
        'Gender': data['Gender'],
        'Category': data['Category'],
        'Purchase Amount (USD)': safe_amount,
        'Review Rating': safe_rating,
        'Previous Purchases': int(data['PreviousPurchases']),
        # Mặc định
        'Season': 'Winter', 
        'Shipping Type': 'Express',
        'Payment Method': 'PayPal', 
        'Frequency of Purchases': 'Weekly'
    }])

    # 4. Feature Engineering (Gọi từ utils)
    input_engineered = apply_feature_engineering(input_df)
    
    # 5. Dự đoán
    prob_yes = model.predict_proba(input_engineered)[0][1]
    prob_no = 1 - prob_yes
    prediction_label = "Yes" if prob_yes >= 0.55 else "No" # Ngưỡng 0.55

    return jsonify({
        'prediction': prediction_label,
        'prob_yes': round(prob_yes * 100, 1),
        'prob_no': round(prob_no * 100, 1),
        'message': "Dữ liệu hợp lệ"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)