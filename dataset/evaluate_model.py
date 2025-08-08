import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import xgboost as xgb
from xgboost import DMatrix

# ----- โหลด Scaler และโมเดล -----
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_gpu_model.json")

# ----- โหลดและเตรียมชุดทดสอบ -----
df = pd.read_csv("UNSW_NB15_testing-set.csv")
assert "label" in df.columns, "ERROR: 'label' column not found in dataset"

# เลือกเฉพาะคอลัมน์ตัวเลข
df = df.select_dtypes(include=["int64", "float64"])

X_test = df.drop("label", axis=1)
y_test = df["label"]

# Scaling
X_test_scaled = scaler.transform(X_test)

# Random Forest Probabilities
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# XGBoost Probabilities
dtest = DMatrix(X_test_scaled)
y_prob_xgb = xgb_model.predict(dtest)

# ----- คำนวณ Calibration Curve -----
prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_prob_rf, n_bins=10)
prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, y_prob_xgb, n_bins=10)

# ----- วาดกราฟ -----
plt.figure(figsize=(6, 6))
plt.plot(prob_pred_rf, prob_true_rf, marker='o', label="RandomForest")
plt.plot(prob_pred_xgb, prob_true_xgb, marker='o', label="XGBoost")
plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")

plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("calibration_curve.png")

