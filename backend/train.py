# train.py
import pandas as pd
import os
import joblib  # Thư viện để lưu model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from utils import apply_feature_engineering # Import hàm từ file utils

# Cấu hình đường dẫn
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'shopping_behavior_updated.csv')
model_path = os.path.join(base_dir, 'model.pkl')

def train_and_save():
    if not os.path.exists(file_path):
        print("Lỗi: Không tìm thấy file csv!")
        return

    print("Đang đọc dữ liệu...")
    df = pd.read_csv(file_path)
    
    # Xử lý target
    target_col = "Subscription Status"
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    y = df[target_col].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Xử lý features
    drop_cols = ["Customer ID", "Item Purchased", "Location", "Size", "Color",
                 "Promo Code Used", "Discount Applied", target_col]
    X_raw = df.drop(columns=drop_cols)
    X_engineered = apply_feature_engineering(X_raw)

    # Định nghĩa pipeline
    numeric_features = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases", "Is_Adult_Male"]
    categorical_features = ["Gender", "Category", "Season", "Age_Group", "Rating_Level", "Customer_Tier"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    print("Đang huấn luyện mô hình...")
    clf.fit(X_engineered, y)
    
    # LƯU MODEL RA FILE
    joblib.dump(clf, model_path)
    print(f"--> Đã lưu model thành công tại: {model_path}")

if __name__ == "__main__":
    train_and_save()