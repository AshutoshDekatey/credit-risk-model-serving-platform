from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

# Synthetic historical loan application
# Features:
# income, debt_ratio, credit_score, missed_payments

X = np.array([
    [90000, 0.15, 780, 0],
    [75000, 0.20, 740, 0],
    [60000, 0.30, 700, 1],
    [45000, 0.40, 650, 2],
    [35000, 0.55, 590, 4],
    [30000, 0.65, 540, 6],
    [120000, 0.10, 820, 0],
    [50000, 0.35, 680, 1],
    [40000, 0.50, 610, 3],
    [25000, 0.70, 520, 7],
])

# 0 = Lower credit risk
# 1 = Higher credit risk

y = np.array([
    0, 0, 0, 0, 1,
    1, 0, 0, 1, 1
])

model = LogisticRegression(max_iter =1000)
model.fit(X,y)

Path("model").mkdir(exist_ok = True)

with open("model/credit_risk_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model artifact created: model/credit_risk_model.pkl")

