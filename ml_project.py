import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Mobile Fraud Detection",
    page_icon="📱",
    layout="centered")


# Load Model

model = joblib.load("model.pkl")

# Sidebar Navigation

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Fraud Prediction", "About"])


# ABOUT PAGE

if page == "About":
    st.title("📘 About This Project")

    st.write("""
    This AI application predicts whether a mobile money transaction 
    is fraudulent or legitimate.

    ### 🔍 Model Used:
    - Decision Tree Classifier
    - Class balancing applied

    ### 📊 Dataset:
    PaySim Mobile Money Fraud Dataset

    ### 🎯 Purpose:
    Help financial institutions detect suspicious transactions.
    """)

    st.info("Developed for Machine Learning Project - 2026")


# PREDICTION PAGE

if page == "Fraud Prediction":

    st.title("📱 Mobile Transaction Fraud Detection System")
    st.write("Enter transaction details below to check fraud risk.")

    
    # User Inputs
    
    step = st.number_input("Step (Time)", min_value=1, value=1)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("Sender Balance After", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver Balance After", min_value=0.0, value=1000.0)

    txn_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    
    # Prediction

    if st.button("🔍 Predict Fraud Risk"):

        input_data = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_CASH_OUT': 0,
            'type_DEBIT': 0,
            'type_PAYMENT': 0,
            'type_TRANSFER': 0
        }

        input_data[f"type_{txn_type}"] = 1

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")

        st.write(f"Fraud Risk Probability: **{round(probability*100, 2)}%**")

        # Risk Level Indicator
        if probability > 0.7:
            st.warning("High Risk Transaction")
        elif probability > 0.4:
            st.info("Moderate Risk Transaction")
        else:
            st.success("Low Risk Transaction")
