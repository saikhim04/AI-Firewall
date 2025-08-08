import pandas as pd
import joblib
from xgboost import Booster, DMatrix
from sklearn.ensemble import RandomForestClassifier

# โหลดโมเดลและ scaler
scaler = joblib.load("scaler.pkl")

# --- Random Forest ---
rf_model = joblib.load("random_forest_model.pkl")

# --- XGBoost ---
xgb_model = Booster()
xgb_model.load_model("xgboost_gpu_model.json")

# โหลดหรือเตรียม dataset จริงที่ต้องการทดสอบ
df_real = pd.read_csv("UNSW_NB15_testing-set.csv")  # หรือข้อมูลจากระบบ packet capture จริง

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข (และต้องตรงกับที่ใช้ตอนเทรน)
df_real = df_real.select_dtypes(include=["int64", "float64"])

# ตรวจสอบว่ามีคอลัมน์ "label" อยู่หรือไม่ (ถ้ามีและไม่ต้องการใช้ ให้ลบทิ้ง)
if "label" in df_real.columns:
    df_real = df_real.drop("label", axis=1)

# ทำ scaling ข้อมูล
X_real_scaled = scaler.transform(df_real)

# ใช้ Random Forest ทำนาย
y_pred_rf = rf_model.predict(X_real_scaled)
print("== Random Forest Prediction ==")
print(y_pred_rf)

# ใช้ XGBoost ทำนาย
dreal = DMatrix(X_real_scaled)
y_pred_xgb = xgb_model.predict(dreal)
y_pred_xgb = (y_pred_xgb > 0.5).astype(int)
print("== XGBoost Prediction ==")
print(y_pred_xgb)

def interpret_label(y):
    return ["allow" if label == 0 else "deny" for label in y]

from collections import Counter

result = interpret_label(y_pred_xgb)  # หรือ y_pred_rf แล้วแต่โมเดล
print(Counter(result))