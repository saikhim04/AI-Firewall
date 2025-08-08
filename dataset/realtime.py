import pandas as pd
import joblib
from xgboost import DMatrix
import numpy as np

# โหลดโมเดลและ scaler
model = joblib.load("xgboost_gpu_model.pkl")
scaler = joblib.load("scaler.pkl")

# สร้าง LabelEncoder แบบกำหนด classes เอง (ให้ตรงกับตอนเทรน)
from sklearn.preprocessing import LabelEncoder

proto_enc = LabelEncoder()
proto_enc.classes_ = np.array(['icmp', 'tcp', 'udp'])

state_enc = LabelEncoder()
state_enc.classes_ = np.array(['ACC', 'CON', 'FIN', 'INT', 'REQ'])

service_enc = LabelEncoder()
service_enc.classes_ = np.array(['-', 'dns', 'ftp', 'http', 'smtp'])

# รับ input
def ask_user_input():
    print("\nป้อนข้อมูล Packet:")
    proto = input("โปรโตคอล (icmp / tcp / udp): ").strip().lower()
    state = input("state (ACC / CON / FIN / INT / REQ): ").strip().upper()
    service = input("service (- / dns / ftp / http / smtp): ").strip().lower()
    ct_state_ttl = int(input("ct_state_ttl (จำนวน state ttl): "))
    ct_srv_dst = int(input("ct_srv_dst (จำนวน connection ไปปลายทาง): "))
    ct_dst_ltm = int(input("ct_dst_ltm (จำนวน connection ไป destination ล่าสุด): "))

    # Encode ข้อมูล
    data = {
        "proto": proto_enc.transform([proto])[0],
        "state": state_enc.transform([state])[0],
        "service": service_enc.transform([service])[0],
        "ct_state_ttl": ct_state_ttl,
        "ct_srv_dst": ct_srv_dst,
        "ct_dst_ltm": ct_dst_ltm
    }

    df = pd.DataFrame([data])

    # scale input
    X_scaled = scaler.transform(df)

    # สร้าง DMatrix สำหรับใช้กับ GPU
    dmatrix = DMatrix(X_scaled)

    # ทำนาย
    y_pred = model.predict(dmatrix)
    result = "ALLOW" if y_pred[0] < 0.5 else "DENY"
    print(f"\nผลการทำนาย: {result} (raw={y_pred[0]:.4f})")

# วนถาม user
while True:
    ask_user_input()
    again = input("\nจะทดสอบ Packet ใหม่หรือไม่? (y/n): ").strip().lower()
    if again != 'y':
        break

