from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Cấu hình đường dẫn
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Định nghĩa đường dẫn tới folder frontend (nằm ngang hàng với backend)
frontend_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend"))

# 2. Kiểm tra xem file index.html có thực sự ở đó không
if not os.path.exists(os.path.join(frontend_dir, 'index.html')):
    # Nếu không thấy ở folder frontend, thử tìm ngay trong folder hiện tại (backend)
    if os.path.exists(os.path.join(base_dir, 'index.html')):
        frontend_dir = base_dir
        print("--> Đã tìm thấy index.html ở cùng thư mục backend.")
    else:
        print(f"--> CẢNH BÁO LỖI: Không tìm thấy file index.html ở cả {frontend_dir} lẫn {base_dir}")

file_path = os.path.join(base_dir, 'shopping_behavior_updated.csv')

# 3. Khởi tạo Flask với đường dẫn template đã tìm được
app = Flask(__name__, template_folder=frontend_dir, static_folder=frontend_dir)
# ==========================================
# 1. FEATURE ENGINEERING NÂNG CAO
# ==========================================
def apply_feature_engineering(df):
    df = df.copy()
    
    # Rời rạc hóa Tuổi
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 101], 
                             labels=['Under_18', '18-30', '31-50', 'Over_50'], right=False)

    # Biến tương tác
    df['Is_Adult_Male'] = ((df['Gender'] == 'Male') & (df['Age'] >= 18)).astype(int)

    # Phân nhóm Đánh giá
    df['Rating_Level'] = pd.cut(df['Review Rating'], bins=[0, 3, 4, 6.0], 
                                labels=['Low_Rating', 'Medium_Rating', 'High_Rating'], right=False)

    # Phân nhóm Khách hàng
    df['Customer_Tier'] = pd.cut(df['Previous Purchases'], bins=[0, 6, 20, 101], 
                                 labels=['New_Customer', 'Regular_Customer', 'VIP_Customer'], right=False)

    return df

# ==========================================
# 2. HUẤN LUYỆN MÔ HÌNH
# ==========================================
def train_model():
    if not os.path.exists(file_path):
        print("Lỗi: Không tìm thấy file csv!")
        return None

    df = pd.read_csv(file_path)
    target_col = "Subscription Status"
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    y = df[target_col].apply(lambda x: 1 if x == 'yes' else 0)
    
    drop_cols = ["Customer ID", "Item Purchased", "Location", "Size", "Color",
                 "Promo Code Used", "Discount Applied", target_col]
    
    # Chỉ giữ lại các cột có trong Feature Engineering để tránh warning
    X_raw = df.drop(columns=drop_cols)
    X_engineered = apply_feature_engineering(X_raw)

    numeric_features = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases", "Is_Adult_Male"]
    categorical_features = ["Gender", "Category", "Season", "Age_Group", "Rating_Level", "Customer_Tier"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # Class_weight balanced là rất quan trọng với bộ dữ liệu lệch này
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    clf.fit(X_engineered, y)
    print("Mô hình đã được huấn luyện xong.")
    return clf

model = train_model()

# ==========================================
# 3. ROUTES
# ==========================================

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
    data = request.json
    
    # 1. Lấy dữ liệu từ Frontend
    raw_age = int(data['Age'])
    raw_rating = float(data['ReviewRating'])
    raw_amount = float(data['PurchaseAmount'])
    
    # 2. Xử lý logic an toàn (Clamping): 
    # Vì dataset train từ 18-70, rating 2.5-5.0, amount 20-100
    # Nếu user nhập ngoài khoảng này, ta đưa về biên gần nhất để mô hình không bị loạn.
    safe_age = max(18, min(raw_age, 70))
    safe_rating = max(2.5, min(raw_rating, 5.0)) 
    safe_amount = max(20, min(raw_amount, 100))

    input_df = pd.DataFrame([{
        'Age': safe_age,
        'Gender': data['Gender'],
        'Category': data['Category'],
        'Purchase Amount (USD)': safe_amount,
        'Review Rating': safe_rating,
        'Previous Purchases': int(data['PreviousPurchases']),
        # Các giá trị mặc định để mô hình chạy được (vì form frontend không hỏi)
        'Season': 'Winter', 
        'Shipping Type': 'Express',
        'Payment Method': 'PayPal', 
        'Frequency of Purchases': 'Weekly'
    }])

    input_engineered = apply_feature_engineering(input_df)
    
    # Lấy xác suất của lớp 1 (Yes)
    prob_yes = model.predict_proba(input_engineered)[0][1]
    prob_no = 1 - prob_yes
    
    # Quyết định nhãn dựa trên ngưỡng xác suất (tùy chỉnh ngưỡng nếu cần)
    prediction_label = "Yes" if prob_yes >= 0.6 else "No" # Nâng ngưỡng nhẹ lên 0.55 để giảm False Positive

    return jsonify({
        'prediction': prediction_label,
        'prob_yes': round(prob_yes * 100, 1),
        'prob_no': round(prob_no * 100, 1),
        'message': "Dữ liệu hợp lệ"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)