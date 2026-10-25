import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer may leave the company.")

@st.cache_resource
def train_model():
    df = pd.read_csv("churn.csv")

    # Clean column names and convert TotalCharges
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService"
    ]

    X = df[features]
    y = df["Churn"].astype(str).str.strip().map(
        {"Yes": 1, "No": 0}
    )

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical = ["Contract", "InternetService"]

    prep = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median")
        ), numeric),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), categorical)
    ])

    model = make_pipeline(
        prep,
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

try:
    model, X_test, y_test = train_model()

    with st.expander("Model Evaluation"):
        predictions = model.predict(X_test)
        report = classification_report(
            y_test, predictions,
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Enter Customer Details")

    with st.form("customer_form"):
        tenure = st.number_input(
            "Tenure (months)", min_value=0,
            max_value=100, value=12
        )

        monthly = st.number_input(
            "Monthly Charges", min_value=0.0,
            value=70.0, step=5.0
        )

        total = st.number_input(
            "Total Charges", min_value=0.0,
            value=840.0, step=10.0
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        submitted = st.form_submit_button(
            "Predict Churn"
        )

    if submitted:
        customer = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet
        }])

        prediction = model.predict(customer)[0]
        probability = model.predict_proba(customer)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer may leave the company.")
        else:
            st.success("✅ Customer is predicted to stay.")

        st.metric(
            "Estimated Churn Probability",
            f"{probability * 100:.1f}%"
        )

        st.caption(
            "This is a machine-learning estimate, not a guarantee."
        )

except FileNotFoundError:
    st.error("churn.csv file was not found. Keep it beside app.py.")
except KeyError as e:
    st.error(
        f"Dataset column missing: {e}. "
        "Check the column names in churn.csv."
    )
except Exception as e:
    st.error(f"Something we
