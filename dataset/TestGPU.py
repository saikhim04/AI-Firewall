import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import cupy as cp
import joblib

# ===== 1. Load dataset =====
df = pd.read_csv("UNSW_NB15_training-set.csv")

# ===== 2. เลือกฟีเจอร์ที่ใช้ =====
features = ['proto', 'state', 'service', 'ct_state_ttl', 'ct_srv_dst', 'ct_dst_ltm']
df = df[features + ['label']]

# ===== 3. เข้ารหัสข้อมูลเชิงหมวดหมู่ =====
encoders = {}
for col in ['proto', 'state', 'service']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ===== 4. เตรียมข้อมูล Train/Test =====
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===== 5. เทรน XGBoost GPU =====
xgb_clf = XGBClassifier(
    tree_method='hist',
    device='cuda',
    objective='binary:logistic',
    eval_metric='logloss'
)
xgb_clf.fit(X_train, y_train)

# ===== 6. Manual Calibration (Platt scaling) =====
# Predict probability บน GPU (train set)
X_train_gpu = cp.array(X_train)
train_probs = xgb_clf.predict_proba(X_train_gpu)[:, 1] # .get() → numpy บน CPU

# ฝึก Logistic Regression เป็นตัว calibrator
calibrator = LogisticRegression()
calibrator.fit(train_probs.reshape(-1, 1), y_train)

# ===== 7. ประเมินผล =====
# Predict test set บน GPU
X_test_gpu = cp.array(X_test)
test_probs = xgb_clf.predict_proba(X_test_gpu)[:, 1]

# Calibrate output
y_prob_calib = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
y_pred_calib = (y_prob_calib > 0.5).astype(int)

print("=== XGBoost (Calibrated) Results ===")
print(classification_report(y_test, y_pred_calib))

# ===== 8. บันทึกโมเดลและ encoder/scaler =====
joblib.dump(xgb_clf, "xgboost_gpu.pkl")
joblib.dump(calibrator, "calibrator.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders['proto'], "encoder_proto.pkl")
joblib.dump(encoders['state'], "encoder_state.pkl")
joblib.dump(encoders['service'], "encoder_service.pkl")

print(xgb_clf.get_booster().attributes())

