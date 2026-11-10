# Deploy Churn Predictor

# ======================================================

# Import Library
import pandas as pd
import numpy as np
import os
import streamlit as st
import pickle
import joblib

# ======================================================

st.set_page_config(
    page_title = "Telco Customer Churn Predictor",
    page_icon = "📡💫",
    layout = "wide")

st.markdown("""
<style>
.result-stay  {background:#EFF6FF; border-left:6px solid #2563EB; color:#1E3A8A;
               padding:1rem 1.2rem; border-radius:8px; font-size:1.3rem; font-weight:600;}
.result-churn {background:#FEF2F2; border-left:6px solid #DC2626; color:#991B1B;
               padding:1rem 1.2rem; border-radius:8px; font-size:1.3rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# Judul Utama
col_title, col_image = st.columns([8,1], vertical_alignment="center")
with col_title:
    st.header("📡 Telco Customer Churn Predictor 📡")
    st.caption(" ⬅️ Input the customer's data in the sidebar to predict the churn probability.")
with col_image:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQK90TP5srqTQScb1yOUDf1lUy5ZStara20nWQ6IRgusRJ9-IsB-xvw5z8&s=10", width=100)
st.divider()


# MEMBUAT HEADER PADA SIDEBAR
st.sidebar.header("Please Input Customer's Data")

def user_input():
    # Input tiap label
    with st.sidebar.form("input_form"):
        # Input label Dependents
        Dependents = st.sidebar.selectbox(label="Dependents",options=['Yes', 'No'])

        # Input label tenure
        tenure = st.sidebar.number_input(label="tenure", min_value=0, max_value=72, value=25)

        # Input label OnlineSecurity
        OnlineSecurity = st.sidebar.selectbox(label="OnlineSecurity",options=['No', 'Yes', 'No internet service'])

        # Input label OnlineBackup
        OnlineBackup = st.sidebar.selectbox(label="OnlineBackup",options=['No', 'Yes', 'No internet service'])

        # Input label InternetService
        InternetService = st.sidebar.selectbox(label="InternetService",options=['DSL', 'Fiber optic', 'No'])

        # Input label DeviceProtection
        DeviceProtection = st.sidebar.selectbox(label="DeviceProtection",options=['No', 'Yes', 'No internet service'])

        # Input label TechSupport
        TechSupport = st.sidebar.selectbox(label="TechSupport",options=['No', 'Yes', 'No internet service'])

        # Input label Contract
        Contract = st.sidebar.selectbox(label="Contract",options=['Month-to-month', 'Two year', 'One year'])

        # Input label PaperlessBilling
        PaperlessBilling = st.sidebar.selectbox(label="PaperlessBilling",options=['No', 'Yes'])

        # Input label MonthlyCharges
        MonthlyCharges=st.sidebar.number_input(label="MonthlyCharges", min_value=18.0 ,max_value=120.0, value=25.0)
        
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)
    # -------------------------------------------------------------------------------------------

    # Simpan dalam DataFrame
    df=pd.DataFrame()
    df['Dependents']=[Dependents]
    df['tenure']=[tenure]
    df['OnlineSecurity']=[OnlineSecurity]
    df['OnlineBackup']=[OnlineBackup]
    df['InternetService']=[InternetService]
    df['DeviceProtection']=[DeviceProtection]
    df['TechSupport']=[TechSupport]
    df['Contract']=[Contract]
    df['PaperlessBilling']=[PaperlessBilling]
    df['MonthlyCharges']=[MonthlyCharges]

    return df, submitted

# Menampilkan user input feature
df_customer, submitted = user_input()

# Import Model Fina LogisticRegression
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'final_model_logreg.sav'))

# Membuat 2 tab : Prediction dan Feature Info
tab_predict, tab_info = st.tabs(["Prediction", "Feature Info"])

# Isi tab Prediction
with tab_predict:
    # Bagi halaman jadi 2 kolom
    column_left, column_right=st.columns(2)
    # Menampilkan dataframe di kolom pertama/kiri
    with column_left:
        st.subheader("📝 Customer's Data 📝")
        st.dataframe(df_customer.astype(str).T.rename(columns={0: "Value"}), width="stretch")

    with column_right:
        st.subheader("🔮 Customer Prediction 🔮")
        if submitted:
            kelas = model.predict(df_customer)
            proba = model.predict_proba(df_customer)[0][1]
            if kelas[0] == 1:
                st.error("⚠️ Customer will CHURN")
            else:
                st.success("✅ Customer will STAY")
            st.write("")
            st.metric("Churn Probability", f"{proba:.1%}")
            st.progress(float(proba))
        else:
            st.info("🪄 Click Predict in the sidebar to view the result 🪄")

# Isi tab Feature Info
with tab_info:
    st.markdown("""
    | Feature | Description |
    |---|---|
    | **Dependents** | Whether the customer has dependents or not. |
    | **tenure** | Number of months the customer has stayed with the company. |
    | **OnlineSecurity** | Whether the customer has online security or not. |
    | **OnlineBackup** | Whether the customer has online backup or not. |
    | **InternetService** | Whether the client is subscribed to Internet service. |
    | **DeviceProtection** | Whether the client has device protection or not. |
    | **TechSupport** | Whether the client has tech support or not. |
    | **Contract** | Type of contract according to duration. |
    | **PaperlessBilling** | Bills issued in paperless form. |
    | **MonthlyCharges** | Amount of charge for service on monthly bases. |
    """)

