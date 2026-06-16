import streamlit as st
import pandas as pd
import pickle
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Churn Prediction", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "home"
    st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 8px;
    background-color: black !important;
    color: white !important;
    border: none;
}

div.stButton > button:hover {
    background-color: #333 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "home"

with col2:
    if st.button("📊 Prediction"):
        st.session_state.page = "prediction"

# ---------- BACKGROUND FUNCTION ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}

    h1, h2, h3, h4, h5, h6, p, label, div {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
with st.spinner("Loading model... please wait ⏳"):
    with open("Customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# ================= HOME PAGE =================
# ================= HOME PAGE =================
if st.session_state.page == "home":

    # convert image to base64
    with open("homepage.png", "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .hero-container {{
        position: relative;
        text-align: center;
        color: white;
    }}
    .hero-text {{
        position: absolute;
        top: 15%;
        left: 50%;
        transform: translate(-50%, -50%);
    }}
    .hero-text h1 {{
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 8px black;
    }}
    .hero-text h4 {{
        font-size: 20px;
        text-shadow: 2px 2px 6px black;
    }}
    </style>

    <div class="hero-container">
        <img src="data:image/png;base64,{encoded_img}" 
     style="width:100%; height:100vh; object-fit: cover;">
        <div class="hero-text">
            <h1>Customer Churn Prediction System</h1>
            <h4>Predict customer behavior and reduce churn using AI 🚀</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
# ================= PREDICTION PAGE =================
elif st.session_state.page == "prediction":

    # ✅ Apply background ONLY here
    set_bg("background.png")

    st.markdown("""
    <h1 style='text-align: center;'>📊 Customer Churn Prediction</h1>
    """, unsafe_allow_html=True)

    st.markdown("### 🧾 Enter Customer Details")

    # ---------- INPUTS ----------
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        senior = st.selectbox("🎂 Senior Citizen", [0, 1])
        partner = st.selectbox("💍 Partner", ["Yes", "No"])
        dependents = st.selectbox("👨‍👩‍👧 Dependents", ["Yes", "No"])
        tenure = st.number_input("📅 Tenure (months)", min_value=0)

        phone = st.selectbox("📞 Phone Service", ["Yes", "No"])
        multiple = st.selectbox("📱 Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("🔒 Online Security", ["Yes", "No"])
        online_backup = st.selectbox("💾 Online Backup", ["Yes", "No"])

    with col2:
        device_protect = st.selectbox("🛡️ Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("🧑‍💻 Tech Support", ["Yes", "No"])
        stream_tv = st.selectbox("📺 Streaming TV", ["Yes", "No"])
        stream_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No"])

        contract = st.selectbox("📜 Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("📄 Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

        monthly = st.number_input("💵 Monthly Charges")
        total = st.number_input("💰 Total Charges")

    st.markdown("---")

    # ---------- BUTTON STYLE ----------
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4FC3F7;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 200px;
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #29B6F6;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- CENTER BUTTON ----------
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_btn = st.button("🔍 Predict Churn")

    # ---------- PREDICTION ----------
    if predict_btn:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protect,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        df = pd.DataFrame([customer_data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_names, fill_value=0)

        prediction = loaded_model.predict(df)

        # ---------- RESULT ----------
        if prediction[0] == 1:
            st.markdown("## ⚠️ High Churn Risk")
            st.error("🚨 This customer is likely to leave the service.")
        else:
            st.markdown("## ✅ Low Churn Risk")
            st.success("🎉 This customer is likely to stay.")
