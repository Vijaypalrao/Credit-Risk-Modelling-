# ==========================================================
# CREDIT RISK PREDICTION WEB APP
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================================
# LOAD SAVED MODEL
# ==========================================================

model = joblib.load("best_credit_risk_model.pkl")

# ==========================================================
# TITLE
# ==========================================================

st.title("🏦 Credit Risk Prediction System")

st.markdown("""
Enter the applicant's details below and click **Predict Loan Risk**
to estimate the probability of loan default.
""")

st.divider()

# ==========================================================
# INPUT FORM
# ==========================================================

with st.form("loan_form"):

    col1, col2 = st.columns(2)

    # ---------------- LEFT COLUMN ----------------

    with col1:

        person_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=32
        )

        person_income = st.number_input(
            "Annual Income",
            min_value=0,
            value=55000
        )

        person_emp_length = st.number_input(
            "Employment Length (Years)",
            min_value=0,
            max_value=60,
            value=6
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=500,
            value=12000
        )

        loan_intent = st.selectbox(
            "Loan Purpose",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

    # ---------------- RIGHT COLUMN ----------------

    with col2:

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=11.5
        )

        loan_percent_income = st.number_input(
            "Loan Percent Income",
            min_value=0.00,
            max_value=1.00,
            value=0.22
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            value=8
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default",
            ["N", "Y"]
        )

    predict_button = st.form_submit_button("Predict Loan Risk")

# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # ---------------------------------------------
    # CREATE INPUT DATAFRAME
    # ---------------------------------------------

    new_customer = pd.DataFrame({

        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "person_home_ownership": [person_home_ownership],
        "loan_amnt": [loan_amnt],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "cb_person_default_on_file": [cb_person_default_on_file]

    })

    # ---------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------

    #prediction = model.predict(new_customer)[0]

    pd_new = model.predict_proba(new_customer)[0, 1]

    # ---------------------------------------------
    # BUSINESS DECISION
    # ---------------------------------------------

    CUTOFF_PD = 0.10

    if pd_new <= CUTOFF_PD:
        decision = "APPROVE LOAN"
    else:
        decision = "REJECT LOAN"

    # ---------------------------------------------
    # EXPECTED LOSS
    # ---------------------------------------------

    LGD = 0.45

    EAD = loan_amnt

    expected_loss = pd_new * LGD * EAD

    # ---------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    #st.success(f"Predicted Class : {prediction}")

    st.info(f"Probability of Default : {pd_new:.2%}")

    st.warning(f"Decision : {decision}")

    st.error(f"Expected Loss : ₹{expected_loss:,.2f}")