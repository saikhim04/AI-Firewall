import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, confusion_matrix
import joblib, json
import csv, time, os
from scapy.all import sniff, IP
from scapy.layers.inet import TCP, UDP

# -----------------------------
# Config
# -----------------------------
NUM_MODEL_LOOPS = 10
MAX_EPOCHS_PER_MODEL = 600
EARLY_STOP_PATIENCE = 5
TARGET_IFACE = "eno2"  # เปลี่ยนเป็น interface ของคุณ

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("ai_firewall_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1).to(device)

# -----------------------------
# Model
# -----------------------------
class FirewallNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Class weight
# -----------------------------
y_np_int = y_train.cpu().numpy().ravel().astype(int)
pos = (y_np_int == 1).sum()
neg = (y_np_int == 0).sum()
ratio = neg / max(pos,1)
pos_weight = torch.tensor([1.2*ratio], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -----------------------------
# Training loop: multiple models
# -----------------------------
best_overall_f1 = 0
best_model_state = None
best_scaler = scaler
best_threshold = 0.5

for model_loop in range(NUM_MODEL_LOOPS):
    print(f"\n=== Model Loop {model_loop+1}/{NUM_MODEL_LOOPS} ===")

    model = FirewallNN(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0
    no_improve = 0

    for epoch in range(MAX_EPOCHS_PER_MODEL):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test)
            probs = torch.sigmoid(logits_test).cpu().numpy().ravel()
        y_pred = (probs > 0.5).astype(int)
        f1 = f1_score(y_test.cpu().numpy(), y_pred)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}, Test F1={f1:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"No improvement for {EARLY_STOP_PATIENCE} epochs, stopping this model.")
                break

    # Pick threshold that maximizes F1
    prec, rec, ths = precision_recall_curve(y_test.cpu().numpy(), probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.nanargmax(f1s)
    THRESH = ths[best_idx] if best_idx < len(ths) else 0.5

    if best_f1 > best_overall_f1:
        best_overall_f1 = best_f1
        best_model_state = model.state_dict()
        best_threshold = THRESH
        best_scaler = scaler
        print(f"New overall best F1={best_overall_f1:.4f}, saved model and threshold={best_threshold:.3f}")

# -----------------------------
# Save best model/scaler/threshold
# -----------------------------
torch.save(best_model_state, "ai_fw_model_best.pt")
joblib.dump(best_scaler, "ai_fw_scaler.pkl")
with open("ai_fw_threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f)
print("✅ Saved best model/scaler/threshold")

# -----------------------------
# Live sniff
# -----------------------------
LOG_FILE = "ai_fw_decisions.csv"

def safe_int(value, default=0):
    try:
        return int(value)
    except:
        return default

def packet_callback(pkt):
    if IP in pkt:
        proto = pkt[IP].proto
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        try:
            srcIP = int.from_bytes(bytes(map(int, src_ip.split("."))), "big")
            dstIP = int.from_bytes(bytes(map(int, dst_ip.split("."))), "big")
        except:
            return

        feat_np = np.array([[proto, sport, dport, srcIP, dstIP]], dtype=np.float32)
        feat_np = np.nan_to_num(feat_np, copy=False, posinf=1e9, neginf=-1e9)
        feat_np = best_scaler.transform(feat_np)
        feat = torch.tensor(feat_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            p = torch.sigmoid(model(feat)).item()
            decision = "ALLOW" if p > best_threshold else "DENY"
            print(f"{decision} (p={p:.3f}, th={best_threshold:.3f})")

        try:
            write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
            with open(LOG_FILE, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["ts","src","dst","proto","sport","dport","prob","thresh","decision"])
                w.writerow([time.time(), src_ip, dst_ip, proto, sport, dport, round(p,3), round(best_threshold,3), decision])
        except Exception as e:
            print(f"[LOG WARN] {e}")

if __name__ == "__main__":
    try:
        sniff(prn=packet_callback, iface=TARGET_IFACE, filter="tcp or udp", timeout=60, store=False)
    except PermissionError:
        print("PermissionError: ต้องรันด้วย sudo หรือ setcap ให้ python ใน venv")
