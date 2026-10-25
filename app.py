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

    required = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "Churn"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df["Churn"] = df["Churn"].astype(str).str.strip()

    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    if df["Churn"].nunique() != 2:
        raise ValueError("Churn column must contain Yes and No values.")

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService"
    ]

    X = df[features]
    y = df["Churn"].map({"No": 0, "Yes": 1})

    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    categorical_features = ["Contract", "InternetService"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model


try:
    model = load_model()

    st.subheader("Enter Customer Details")

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0, step=5.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0, step=10.0
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    if st.button("Predict Churn", type="primary"):
        customer = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet
        }])

        prediction = model.predict(customer)[0]
        probability = model.predict_proba(customer)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Customer may leave the company.")
        else:
            st.success("Customer is predicted to stay.")

        st.metric(
            "Estimated Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This is a prediction, not a guarantee."
        )

except FileNotFoundError:
    st.error(
        "churn.csv file not found. Upload churn.csv "
        "into the same folder as app.py."
    )
except Exception as error:
    st.error(f"Unable to run the project: {error}")
