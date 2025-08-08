from xgboost import XGBClassifier

model = XGBClassifier(tree_method="gpu_hist")
model.fit([[1,2],[3,4]], [0,1])  # ทดสอบเทรนเล็กๆ

print("XGBoost GPU ทำงานได้")
