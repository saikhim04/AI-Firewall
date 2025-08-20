import os, time, json, csv, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, classification_report, confusion_matrix, f1_score
)

from scapy.all import sniff, IP
from scapy.layers.inet import TCP, UDP

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# 1) Load & split data
# -----------------------------
df = pd.read_csv("ai_firewall_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale with train only
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np  = scaler.transform(X_test_np)

# -----------------------------
# 2) Torch tensors (CPU first)
# -----------------------------
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32).view(-1, 1)

# -----------------------------
# 3) Model
# -----------------------------
class FirewallNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # no sigmoid; we use BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

model = FirewallNN(in_features=X_train.shape[1])

# -----------------------------
# 4) Class weight (on CPU now)
# -----------------------------
y_train_np_int = y_train.numpy().ravel().astype(int)
pos = (y_train_np_int == 1).sum()  # ALLOW
neg = (y_train_np_int == 0).sum()  # DENY
ratio = neg / max(pos, 1)
pos_weight = torch.tensor([0.8 * ratio], dtype=torch.float32)

# -----------------------------
# 5) Device & move everything
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test  = X_test.to(device)
y_test  = y_test.to(device)
pos_weight = pos_weight.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# -----------------------------
# 6) Train
# -----------------------------
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# 7) Evaluate + best threshold
# -----------------------------
model.eval()
with torch.no_grad():
    logits_test = model(X_test)
    probs = torch.sigmoid(logits_test).detach().cpu().numpy().ravel()

y_true_np = y_test.detach().cpu().numpy().astype(int).ravel()
prec, rec, ths = precision_recall_curve(y_true_np, probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1s))
THRESH = float(ths[best_idx]) if best_idx < len(ths) else 0.5

print(f"Best threshold≈{THRESH:.3f}, "
      f"P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}, F1={f1s[best_idx]:.3f}")

y_pred_np = (probs > THRESH).astype(int)
print(classification_report(y_true_np, y_pred_np, digits=4))
cm = confusion_matrix(y_true_np, y_pred_np)
print("Confusion matrix:\n", cm)
print("F1 (best_th):", f1_score(y_true_np, y_pred_np))

# -----------------------------
# เพิ่มเติม: หา threshold ที่ Precision >= 0.8
# -----------------------------
TARGET_P = 0.80  # กำหนดค่า Precision ที่อยากได้

valid_idx = np.arange(1, len(prec))        # เพราะ len(ths) = len(prec)-1
prec_v = prec[valid_idx]
rec_v  = rec[valid_idx]
ths_v  = ths[valid_idx - 1]

mask = prec_v >= TARGET_P
if np.any(mask):
    cand_idx = valid_idx[mask]
    f1_v = 2 * (prec[cand_idx] * rec[cand_idx]) / (prec[cand_idx] + rec[cand_idx] + 1e-9)
    best_cand_rel = int(np.nanargmax(f1_v))
    best_cand_idx = cand_idx[best_cand_rel]
    thr_p = ths[best_cand_idx - 1]

    print(f"\n[Precision>={TARGET_P:.2f}] Picked threshold≈{thr_p:.3f}, "
          f"P={prec[best_cand_idx]:.3f}, R={rec[best_cand_idx]:.3f}, "
          f"F1={f1_v[best_cand_rel]:.3f}")

    y_pred_p = (probs > thr_p).astype(int)
    print(classification_report(y_true_np, y_pred_p, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true_np, y_pred_p))
    print("F1 (P>=target thr):", f1_score(y_true_np, y_pred_p))
else:
    print(f"\n⚠️ ไม่มี threshold ที่ให้ Precision >= {TARGET_P:.2f} บนชุดนี้")

# -----------------------------
# 😎 Save (once)
# -----------------------------
torch.save(model.state_dict(), "ai_fw_model.pt")
joblib.dump(scaler, "ai_fw_scaler.pkl")
with open("ai_fw_threshold.json", "w") as f:
    json.dump({"threshold": float(THRESH)}, f)

print("✅ Saved model/scaler/threshold:", THRESH)
print(f"DENY (0) = {neg}, ALLOW (1) = {pos}")

# -----------------------------
# 9) Live sniff (inference)
# -----------------------------
LOG_FILE = "ai_fw_decisions.csv"

def packet_callback(pkt):
    if IP not in pkt:
        return

    proto = pkt[IP].proto
    sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst

    # meta เพิ่มเติมเพื่อ log
    pkt_len = len(bytes(pkt)) if hasattr(pkt, '_bytes_') else getattr(pkt, 'len', 0)
    ttl = getattr(pkt[IP], 'ttl', None)
    flags = None
    if TCP in pkt:
        # แปลง flags เป็น string เช่น "S", "SA", "PA", ...
        flags = str(pkt[TCP].flags)

    # แปลง IPv4 เป็น int (ใช้ต่อในฟีเจอร์)
    try:
        srcIP = int.from_bytes(bytes(map(int, src_ip.split("."))), "big")
        dstIP = int.from_bytes(bytes(map(int, dst_ip.split("."))), "big")
    except Exception:
        return

    # ฟีเจอร์สำหรับโมเดล
    feat_np = np.array([[proto, sport, dport, srcIP, dstIP]], dtype=np.float32)
    feat_np = np.nan_to_num(feat_np, copy=False, posinf=1e9, neginf=-1e9)
    feat_np = scaler.transform(feat_np)
    feat = torch.tensor(feat_np, dtype=torch.float32, device=device)

    # infer
    with torch.no_grad():
        p = torch.sigmoid(model(feat)).item()
        decision = "ALLOW" if p > THRESH else "DENY"

    # ชื่อโปรโตคอลอ่านง่าย
    proto_name = "TCP" if proto == 6 else ("UDP" if proto == 17 else str(proto))

    # เวลาอ่านง่าย (ISO UTC) + epoch เดิม
    ts_epoch = time.time()
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_epoch))

    # เขียนลง CSV ด้วย DictWriter (ป้องกันลำดับคอลัมน์สลับ)
    header = [
        "ts","ts_iso","iface","src","dst","proto","proto_name",
        "sport","dport","len","ttl","flags",
        "prob","thresh","decision"
    ]
    row = {
        "ts": round(ts_epoch,3),
        "ts_iso": ts_iso,
        "iface": "eno2",           # หรือดึงอัตโนมัติถ้าต้องการ
        "src": src_ip,
        "dst": dst_ip,
        "proto": proto,
        "proto_name": proto_name,
        "sport": sport,
        "dport": dport,
        "len": pkt_len,
        "ttl": ttl,
        "flags": flags,
        "prob": round(p, 3),
        "thresh": round(THRESH, 3),
        "decision": decision
    }

    try:
        write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[LOG WARN] {e}")

    # console print สั้น ๆ
    print(f"{decision} {src_ip}:{sport} -> {dst_ip}:{dport} "
          f"({proto_name}) p={p:.3f} th={THRESH:.3f} len={pkt_len} flags={flags}")

if __name__ == "__main__":
    try:
        # ปรับ filter ให้ชัดและค้างดักต่อเนื่อง (เอา timeout ออกถ้าต้องการ)
        sniff(prn=packet_callback, iface="eno2", filter="ip and (tcp or udp)", timeout=60, store=False)
    except PermissionError:
        print("PermissionError: ต้องรันด้วย sudo หรือ setcap ให้ python ใน venv")