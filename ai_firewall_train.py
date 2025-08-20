import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scapy.all import sniff, IP
from sklearn.metrics import f1_score, precision_recall_curve
import joblib, json, csv, time, os
import csv, time, os
from collections import Counter
from scapy.layers.inet import TCP, UDP
from sklearn.metrics import confusion_matrix, classification_report

# 1) Load data
df = pd.read_csv("ai_firewall_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# 2) Split ก่อน แล้วค่อย fit scaler ด้วย train set เท่านั้น
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3) Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)


# 4) Model (ไม่ใส่ Sigmoid — จะใช้ BCEWithLogitsLoss)
class FirewallNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # <- no Sigmoid here
        )

    def forward(self, x):
        return self.net(x)

model = FirewallNN(in_features=X_train.shape[1])
# คำนวณ weight ของ class (ใช้ข้อมูล y_train ที่เป็น tensor)
y_np = y_train.numpy().ravel().astype(int)
pos = (y_np == 1).sum()  # จำนวนตัวอย่าง class 1 (ALLOW)
neg = (y_np == 0).sum()  # จำนวนตัวอย่าง class 0 (DENY)

# pos_weight = สัดส่วนที่ทำให้ class 1 มีน้ำหนักมากขึ้น
ratio = neg / max(pos, 1)
pos_weight = torch.tensor([1.2 * ratio], dtype=torch.float32) # จาก 1.0*ratio --> 0.8*ratio

# ใช้ BCEWithLogitsLoss แทน BCELoss (เสถียรกว่า)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 5) Train
for epoch in range(600):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6) Evaluate + pick best threshold + report + save (single clean block)
model.eval()
with torch.no_grad():
    # predict prob on test
    logits_test = model(X_test)
    probs = torch.sigmoid(logits_test).cpu().numpy().ravel()

# y_true as numpy
y_true_np = y_test.cpu().numpy().astype(int).ravel()

# find best threshold by F1
prec, rec, ths = precision_recall_curve(y_true_np, probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1s))
THRESH = float(ths[best_idx]) if best_idx < len(ths) else 0.5

print(f"Best threshold≈{THRESH:.3f}, P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}, F1={f1s[best_idx]:.3f}")

# final hard predictions with best threshold
y_pred_np = (probs > THRESH).astype(int)

# metrics
print(classification_report(y_true_np, y_pred_np, digits=4))

# (optional) confusion matrix
cm = confusion_matrix(y_true_np, y_pred_np)
print("Confusion matrix:\n", cm)

# save once
torch.save(model.state_dict(), "ai_fw_model.pt")
joblib.dump(scaler, "ai_fw_scaler.pkl")
with open("ai_fw_threshold.json", "w") as f:
    json.dump({"threshold": float(THRESH)}, f)

print("Saved model/scaler/threshold:", THRESH)

# 7) Live sniff (แยกไว้ — ต้องสิทธิ์ sudo/setcap)
def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def packet_callback(pkt):
    if IP in pkt:
        proto = pkt[IP].proto

        # ดึงพอร์ตจากชั้น TCP/UDP จริงเพื่อลดเคส sport/dport=0
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        try:
            srcIP = int.from_bytes(bytes(map(int, src_ip.split("."))), "big")
            dstIP = int.from_bytes(bytes(map(int, dst_ip.split("."))), "big")
        except Exception:
            return  # ข้ามแพ็กเก็ตที่รูปแบบ IP แปลก

        # 👉 เตรียมฟีเจอร์เป็น numpy ก่อน
        feat_np = np.array([[proto, sport, dport, srcIP, dstIP]], dtype=np.float32)
        feat_np = np.nan_to_num(feat_np, copy=False, posinf=1e9, neginf=-1e9)
        feat_np = scaler.transform(feat_np)

        feat = torch.tensor(feat_np, dtype=torch.float32)

        # infer
        with torch.no_grad():
            p = torch.sigmoid(model(feat)).item()
            decision = "ALLOW" if p > THRESH else "DENY"
            print(f"{decision} (p={p:.3f}, th={THRESH:.3f})")

        # log ลงไฟล์ (เขียน header เมื่อไฟล์ยังว่าง)
        try:
            write_header = not os.path.exists("ai_fw_decisions.csv") or os.path.getsize("ai_fw_decisions.csv") == 0
            with open("ai_fw_decisions.csv", "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["ts","src","dst","proto","sport","dport","prob","thresh","decision"])
                w.writerow([time.time(), src_ip, dst_ip, proto, sport, dport, round(p,3), round(THRESH,3), decision])
        except Exception as e:
            print(f"[LOG WARN] {e}")

# ====== หาค่า threshold ที่ให้ F1 ดีที่สุด ======
probs_np = probs.ravel()   # <-- แปลงเป็น numpy 1D
prec, rec, thr = precision_recall_curve(y_true_np, probs_np)
f1s = 2 * prec * rec / (prec + rec + 1e-12)

ix = np.nanargmax(f1s)
best_thr = thr[ix] if ix < len(thr) else 0.5
print(f"Best threshold≈{best_thr:.3f}, P={prec[ix]:.3f}, R={rec[ix]:.3f}, F1={f1s[ix]:.3f}")

THRESH = float(best_thr)
y_pred_new = (probs_np > THRESH).astype(int)
print("F1 (best_th):", f1_score(y_true_np, y_pred_new))

# หลังจากได้ best_thr แล้ว
THRESH = float(best_thr)

torch.save(model.state_dict(), "ai_fw_model.pt")
joblib.dump(scaler, "ai_fw_scaler.pkl")
with open("ai_fw_threshold.json", "w") as f:
    json.dump({"threshold": THRESH}, f)

print("✅ Saved model/scaler/threshold:", THRESH)


# ====== Save model / scaler / threshold ======
torch.save(model.state_dict(), "ai_fw_model.pt")
joblib.dump(scaler, "ai_fw_scaler.pkl")
json.dump({"threshold": float(THRESH)}, open("ai_fw_threshold.json", "w"))


if __name__ == "__main__":
    try:
        # เปลี่ยน iface ให้ตรงเครื่อง เช่น "eth0" หรือ "wlan0"
        sniff(prn=packet_callback, iface="eno2", filter="tcp or udp", timeout=30, store=False)
        # หมายเหตุ:
        # - filter="ip" จะช่วยลด noise (เลือกเฉพาะ IPv4); จะใช้ "tcp" / "udp" ก็ได้
        # - timeout=30 กันค้าง ถ้าอยากกำหนดจำนวนแพ็กเก็ต ใช้ count=10 ร่วมได้
    except PermissionError:
        print("PermissionError: ต้องรันด้วย sudo หรือ setcap ให้ python ใน venv")

print(f"DENY (0) = {neg}, ALLOW (1) = {pos}") #count ALLOW & DENY





