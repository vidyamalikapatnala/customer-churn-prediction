import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer may leave the company.")

@st.cache_resource
def load_model():
    df = pd.read_csv("churn.csv")
    df.columns = df.columns.str.strip()

    # TotalCharges lo unna space / empty values ni number ga marchadam
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    X = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]]
    y = df["Churn"].map({"Yes": 1, "No": 0})
    y = y.dropna()
    X = X.loc[y.index]

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["Contract", "InternetService"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ churn.csv file not found! GitHub lo churn.csv upload chey.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ Customer may CHURN! Probability: {proba[1]*100:.2f}%")
    else:
        st.success(f"✅ Customer will STAY. Probability: {proba[0]*100:.2f}%")
