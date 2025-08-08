import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from xgboost import DMatrix

# ----- โหลดและเตรียมข้อมูล -----
df = pd.read_csv("UNSW_NB15_training-set.csv")

# ใช้เฉพาะคอลัมน์ที่เป็นตัวเลข
df = df.select_dtypes(include=["int64", "float64"])

# แยก features กับ label
X = df.drop("label", axis=1)
y = df["label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งชุด train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----- เทรน Random Forest -----
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# ทำนายและรายงานผล
y_pred_rf = rf_model.predict(X_test)
print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf))

# บันทึกโมเดล Random Forest
joblib.dump(rf_model, "random_forest_model.pkl")

# ----- เทรน XGBoost (GPU) ผ่าน DMatrix -----
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

params = {
    "tree_method": "hist",         # GPU ต้องใช้ hist
    "device": "cuda",              # ใช้ GPU
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# ทำนาย
y_pred_xgb = xgb_model.predict(dtest)
y_pred_xgb = (y_pred_xgb > 0.5).astype(int)

# รายงานผล
print("=== XGBoost (GPU) Results ===")
print(classification_report(y_test, y_pred_xgb))

# บันทึกโมเดล XGBoost ด้วย joblib (ใช้ booster บันทึก)
xgb_model.save_model("xgboost_gpu_model.json")

# ----- บันทึก Scaler ร่วมด้วย -----
joblib.dump(scaler, "scaler.pkl")