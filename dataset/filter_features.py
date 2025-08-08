import pandas as pd

# โหลดชุด training อย่างเดียวก่อน (เร็ว)
df = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)

# ✅ ฟีเจอร์ที่ Packet Filtering สนใจ
firewall_features = [
    'proto',         # Protocol เช่น TCP, UDP, ICMP
    'state',         # สถานะของ connection เช่น CON, FIN
    'service',       # ประเภทของบริการ เช่น http, ftp
    'ct_state_ttl',  # จำนวน connection ที่มี state/ttl เดียวกัน
    'ct_srv_dst',    # จำนวน connection ที่มี service และ destination เหมือนกัน
    'ct_dst_ltm',    # จำนวน connection ที่ destination เดียวกันในช่วงเวลาล่าสุด
    'label'          # 0 = Normal, 1 = Attack
]

# ✅ ตรวจสอบว่าฟีเจอร์มีอยู่จริงใน dataset
available = [f for f in firewall_features if f in df.columns]

# ✅ ตัดเฉพาะคอลัมน์ที่ต้องการ
df_filtered = df[available]

# ✅ แสดงผล
print("✅ ขนาดข้อมูลหลังเลือกเฉพาะ header-level features:", df_filtered.shape)
print("\n📌 คอลัมน์ที่ใช้:")
print(df_filtered.columns.tolist())

print("\n🔍 ตัวอย่างข้อมูล:")
print(df_filtered.head())