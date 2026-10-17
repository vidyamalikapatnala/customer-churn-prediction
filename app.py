import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

st.write(
    "Welcome! This website helps predict "
    "whether a customer may leave a telecom company."
)

st.divider()

st.header("👤 Enter Customer Details")

tenure = st.number_input(
    "Customer Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    max_value=1000.0,
    value=50.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

if st.button("🔍 Predict Churn"):

    st.subheader("📋 Prediction Result")

    # Demo logic only; not a trained ML model
    risk_score = 0

    if tenure < 12:
        risk_score += 1

    if monthly_charges > 70:
        risk_score += 1

    if contract == "Month-to-month":
        risk_score += 1

    if risk_score >= 2:
        st.warning("⚠️ Demo Result: High Churn Risk")
    else:
        st.success("✅ Demo Result: Low Churn Risk")

st.divider()

st.caption(
    "Student Project | Customer Churn Prediction "
    "using Machine Learning"
)
