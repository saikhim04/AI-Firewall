import pandas as pd

# โหลด training และ testing set
train_df = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)
test_df = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)

# รวมเป็น dataframe เดียว (ถ้าต้องการ)
df = pd.concat([train_df, test_df], ignore_index=True)

# ตรวจสอบข้อมูล
print("✅ ขนาดข้อมูลรวม:", df.shape)
print("\n📌 คอลัมน์ทั้งหมด:")
print(df.columns.tolist())

print("\n🔍 ตัวอย่างข้อมูล:")
print(df.head())

print("\n📊 ค่าของ label (0=normal, 1=attack):")
print(df['label'].value_counts())

# ถ้ามี 'attack_cat' ดูประเภทของการโจมตี
if 'attack_cat' in df.columns:
    print("\n📊 ประเภทการโจมตี:")
    print(df['attack_cat'].value_counts())

