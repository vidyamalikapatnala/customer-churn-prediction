import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("churn.csv")
X = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]]
y = df["Churn"]
prep = ColumnTransformer([
 ("num", SimpleImputer(strategy="median"), ["tenure", "MonthlyCharges", "TotalCharges"]),
 ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), ["Contract", "InternetService"])
])
model = make_pipeline(prep, RandomForestClassifier(random_state=42))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), zero_division=0))
sample = pd.DataFrame([{"tenure": int(input("Tenure months: ")),
 "MonthlyCharges": float(input("Monthly charges: ")), "TotalCharges": float(input("Total charges: ")),
 "Contract": input("Contract (Month-to-month/One year/Two year): "),
 "InternetService": input("Internet (DSL/Fiber optic/No): ")}])
print("Predicted churn:", model.predict(sample)[0])
