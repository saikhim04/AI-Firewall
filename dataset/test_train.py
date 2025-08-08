import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. โหลดข้อมูล
df = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)

# 2. เลือกเฉพาะฟีเจอร์ที่เกี่ยวข้องกับ firewall
features = ['proto', 'state', 'service', 'ct_state_ttl', 'ct_srv_dst', 'ct_dst_ltm']
df = df[features + ['label']]

# 3. สร้างคอลัมน์ action (0 = Allow, 1 = Deny)
df['action'] = df['label'].apply(lambda x: 'Allow' if x == 0 else 'Deny')
df.drop('label', axis=1, inplace=True)

# 4. แปลงข้อมูลประเภท object (เช่น proto) ให้เป็นตัวเลข
for col in ['proto', 'state', 'service']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 5. เตรียมข้อมูล train/test
X = df.drop('action', axis=1)
y = df['action']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. เทรนโมเดล
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# 8. บันทึกโมเดลไว้ใช้ใน realtime.py
import joblib
joblib.dump(clf, "firewall_model.pkl")
# 7. ประเมินผล
y_pred = clf.predict(X_test)
print("Accuracy Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

