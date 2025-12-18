# utils.py
import pandas as pd
import numpy as np

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