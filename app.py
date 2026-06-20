import os
import pickle

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Column order the model was trained on. Must match train_model.py exactly.
FEATURE_NAMES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up",
]

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root{
        --ink:#16262E;
        --paper:#F3F6F5;
        --primary:#0E5C6B;
        --primary-dark:#093C46;
        --coral:#E4572E;
        --safe:#1E8A6E;
        --risk:#C8403D;
        --card:#FFFFFF;
        --muted:#5B6B6B;
        --line:#DDE5E3;
    }

    html, body, [class*="css"]{
        font-family:'Inter', sans-serif;
        color:var(--ink);
    }
    .stApp{ background:var(--paper); }

    h1, h2, h3, .display{
        font-family:'Fraunces', serif !important;
        color:var(--primary-dark);
        letter-spacing:-0.01em;
    }

    /* Hide default Streamlit chrome we don't need */
    #MainMenu, footer{visibility:hidden;}

    /* ---- Hero ---- */
    .hero{
        padding:2.4rem 2.4rem 1.6rem 2.4rem;
        background:linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius:18px;
        color:#F3F6F5;
        margin-bottom:1.4rem;
    }
    .hero h1{ color:#FFFFFF !important; font-size:2.4rem; margin-bottom:0.4rem;}
    .hero p{ color:#D7E6E3; font-size:1.05rem; max-width:640px; line-height:1.55;}

    /* ---- ECG divider ---- */
    .ecg-wrap{ margin: 0.4rem 0 1.6rem 0; }
    .ecg-wrap svg{ width:100%; height:36px; display:block; }

    /* ---- Section card ---- */
    .section-card{
        background:var(--card);
        border:1px solid var(--line);
        border-radius:16px;
        padding:1.5rem 1.7rem;
        margin-bottom:1.3rem;
    }
    .section-title{
        font-family:'Fraunces', serif;
        font-size:1.25rem;
        color:var(--primary-dark);
        margin-bottom:0.9rem;
        display:flex; align-items:center; gap:0.5rem;
    }

    /* ---- Buttons ---- */
    div[data-testid="stButton"] button{
        background:var(--primary);
        color:#fff;
        border:none;
        border-radius:10px;
        padding:0.6rem 1.4rem;
        font-weight:600;
        transition:background 0.15s ease;
    }
    div[data-testid="stButton"] button:hover{
        background:var(--primary-dark);
        color:#fff;
    }

    /* ---- Inputs ---- */
    div[data-baseweb="select"] > div, .stNumberInput input{
        border-radius:8px !important;
        border-color:var(--line) !important;
    }

    /* ---- Result cards ---- */
    .result-card{
        border-radius:18px;
        padding:2rem 2.2rem;
        color:#fff;
        margin-bottom:1.4rem;
    }
    .result-card.safe{ background:linear-gradient(135deg, #1E8A6E 0%, #156653 100%); }
    .result-card.risk{ background:linear-gradient(135deg, #C8403D 0%, #8F2B29 100%); }
    .result-card h2{ color:#fff !important; margin-bottom:0.2rem;}
    .result-card .status-pill{
        display:inline-block;
        background:rgba(255,255,255,0.18);
        padding:0.25rem 0.8rem;
        border-radius:999px;
        font-size:0.8rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
        margin-bottom:0.8rem;
    }
    .result-card ul{ margin-top:0.8rem; padding-left:1.1rem; line-height:1.7;}

    /* ---- Risk meter ---- */
    .meter-label{ display:flex; justify-content:space-between; font-size:0.85rem; color:var(--muted); margin-bottom:0.3rem;}
    .meter-track{
        position:relative;
        width:100%;
        height:14px;
        border-radius:999px;
        background:linear-gradient(90deg, #1E8A6E 0%, #E8C547 50%, #C8403D 100%);
    }
    .meter-marker{
        position:absolute;
        top:-7px;
        width:28px; height:28px;
        border-radius:50%;
        background:#16262E;
        border:3px solid #fff;
        box-shadow:0 2px 6px rgba(0,0,0,0.3);
        transform:translateX(-50%);
    }
    .meter-pct{
        text-align:center;
        margin-top:0.6rem;
        font-family:'Fraunces', serif;
        font-size:1.6rem;
        color:var(--ink);
    }

    /* ---- Footer ---- */
    .app-footer{
        text-align:center;
        padding:1.2rem 0 0.4rem 0;
        color:var(--muted);
        font-size:0.85rem;
        border-top:1px solid var(--line);
        margin-top:2rem;
    }
    .disclaimer{
        font-size:0.8rem;
        color:var(--muted);
        background:#EDEFEA;
        border-left:3px solid var(--primary);
        padding:0.7rem 1rem;
        border-radius:6px;
        margin-top:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ECG_DIVIDER = """
<div class="ecg-wrap">
<svg viewBox="0 0 600 40" preserveAspectRatio="none">
  <polyline points="0,20 120,20 145,20 158,4 172,36 186,20 220,20 240,20 255,8 268,32 282,20 600,20"
            fill="none" stroke="#E4572E" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>
"""

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ `model.pkl` not found. Place it in the same folder as `app.py` "
        "(see train_model.py to regenerate it from heart.csv)."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "patient" not in st.session_state:
    st.session_state.patient = {}
if "result" not in st.session_state:
    st.session_state.result = None


def render_footer():
    st.markdown(
        '<div class="app-footer">Developed by <strong>Raheela Nisar</strong></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGE_LABELS = {"home": "🏠 Home", "details": "📋 Patient Details", "result": "📊 Prediction Result"}
labels = list(PAGE_LABELS.values())
keys = list(PAGE_LABELS.keys())

with st.sidebar:
    st.markdown("### ❤️ Heart Health")
    selected_label = st.radio("Navigate", labels, index=keys.index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = keys[labels.index(selected_label)]
    st.markdown("---")
    st.caption(
        "This app estimates heart disease risk from clinical inputs using a "
        "machine learning model trained on the Heart Failure Prediction "
        "dataset. It is **not** a medical diagnosis."
    )

# ----------------------------------------------------------------------------
# PAGE 1 — HOME
# ----------------------------------------------------------------------------
if st.session_state.page == "home":
    st.markdown(
        """
        <div class="hero">
            <h1>❤️ Heart Disease Prediction System</h1>
            <p>Enter a patient's clinical details and get an instant, data-driven
            estimate of heart disease risk — built on logistic regression trained
            over 900+ real patient records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(ECG_DIVIDER, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="section-card"><div class="section-title">📋 Step 1</div>'
            "Enter the patient's personal and medical details on the next page.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="section-card"><div class="section-title">🧠 Step 2</div>'
            "Our trained model analyzes the inputs against known risk patterns.</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="section-card"><div class="section-title">📊 Step 3</div>'
            "View a clear risk result with a visual meter and recommendations.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("Start Prediction →", use_container_width=False):
        st.session_state.page = "details"
        st.rerun()

    st.markdown(
        '<div class="disclaimer">⚕️ This tool is for educational and informational '
        "purposes only and does not replace professional medical advice, diagnosis, "
        "or treatment. Always consult a qualified cardiologist.</div>",
        unsafe_allow_html=True,
    )
    render_footer()

# ----------------------------------------------------------------------------
# PAGE 2 — PATIENT DETAILS
# ----------------------------------------------------------------------------
elif st.session_state.page == "details":
    st.markdown("## 📋 Patient Details")
    st.markdown(ECG_DIVIDER, unsafe_allow_html=True)

    p = st.session_state.patient

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧍 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("🎂 Age", min_value=1, max_value=120, value=p.get("age", 45))
    with c2:
        sex = st.selectbox("⚧ Sex", ["M", "F"], index=["M", "F"].index(p.get("sex", "M")))
    with c3:
        chestpain = st.selectbox(
            "💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
            index=["ATA", "NAP", "TA", "ASY"].index(p.get("chestpain", "ATA")),
            help="ATA: Atypical Angina · NAP: Non-Anginal Pain · TA: Typical Angina · ASY: Asymptomatic",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Medical Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        restingBP = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=60, max_value=250, value=p.get("restingBP", 120))
        fastingBS = st.selectbox(
            "🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1],
            index=[0, 1].index(p.get("fastingBS", 0)),
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with c2:
        cholesterol = st.number_input("🧪 Cholesterol (mm/dl)", min_value=0, max_value=700, value=p.get("cholesterol", 200))
        restecg = st.selectbox(
            "💓 Resting ECG", ["Normal", "ST", "LVH"],
            index=["Normal", "ST", "LVH"].index(p.get("restecg", "Normal")),
        )
    with c3:
        maxHR = st.number_input("🏃 Max Heart Rate", min_value=60, max_value=220, value=p.get("maxHR", 150))
        angina = st.selectbox(
            "⚡ Exercise-Induced Angina", ["Y", "N"],
            index=["Y", "N"].index(p.get("angina", "N")),
        )

    c1, c2 = st.columns(2)
    with c1:
        oldpeak = st.number_input("📉 Oldpeak (ST depression)", min_value=-2.0, max_value=7.0, value=p.get("oldpeak", 0.0), step=0.1)
    with c2:
        slope = st.selectbox(
            "📈 ST Slope", ["Flat", "Up", "Down"],
            index=["Flat", "Up", "Down"].index(p.get("slope", "Up")),
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict"):
        st.session_state.patient = {
            "age": age, "sex": sex, "chestpain": chestpain,
            "restingBP": restingBP, "cholesterol": cholesterol, "fastingBS": fastingBS,
            "restecg": restecg, "maxHR": maxHR, "angina": angina,
            "oldpeak": oldpeak, "slope": slope,
        }

        row = {
            "Age": age, "RestingBP": restingBP, "Cholesterol": cholesterol,
            "FastingBS": fastingBS, "MaxHR": maxHR, "Oldpeak": oldpeak,
            "Sex_M": 1 if sex == "M" else 0,
            "ChestPainType_ATA": 1 if chestpain == "ATA" else 0,
            "ChestPainType_NAP": 1 if chestpain == "NAP" else 0,
            "ChestPainType_TA": 1 if chestpain == "TA" else 0,
            "RestingECG_Normal": 1 if restecg == "Normal" else 0,
            "RestingECG_ST": 1 if restecg == "ST" else 0,
            "ExerciseAngina_Y": 1 if angina == "Y" else 0,
            "ST_Slope_Flat": 1 if slope == "Flat" else 0,
            "ST_Slope_Up": 1 if slope == "Up" else 0,
        }
        X = pd.DataFrame([row])[FEATURE_NAMES]

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # probability of class 1 (disease)

        st.session_state.result = {"prediction": int(prediction), "probability": float(probability)}
        st.session_state.page = "result"
        st.rerun()

    render_footer()

# ----------------------------------------------------------------------------
# PAGE 3 — PREDICTION RESULT
# ----------------------------------------------------------------------------
elif st.session_state.page == "result":
    st.markdown("## 📊 Prediction Result")
    st.markdown(ECG_DIVIDER, unsafe_allow_html=True)

    result = st.session_state.result
    if result is None:
        st.info("No prediction yet — fill in the patient details first.")
        if st.button("📋 Go to Patient Details"):
            st.session_state.page = "details"
            st.rerun()
    else:
        risk_pct = round(result["probability"] * 100, 1)
        is_high_risk = result["prediction"] == 1

        if is_high_risk:
            st.markdown(
                f"""
                <div class="result-card risk">
                    <span class="status-pill">Risk Status: High</span>
                    <h2>⚠️ High Chance of Heart Disease</h2>
                    <p>The model estimates a <strong>{risk_pct}%</strong> likelihood of heart disease based on the entered details.</p>
                    <ul>
                        <li>Please consult a cardiologist for a thorough evaluation.</li>
                        <li>Maintain a heart-healthy, low-sodium, low-fat diet.</li>
                        <li>Avoid smoking and limit alcohol intake.</li>
                        <li>Schedule regular checkups and monitor blood pressure and cholesterol.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card safe">
                    <span class="status-pill">Risk Status: Low</span>
                    <h2>✅ Low Chance of Heart Disease</h2>
                    <p>The model estimates a <strong>{risk_pct}%</strong> likelihood of heart disease based on the entered details.</p>
                    <ul>
                        <li>Keep up a healthy, balanced diet.</li>
                        <li>Exercise regularly — aim for at least 150 minutes a week.</li>
                        <li>Continue routine health checkups.</li>
                        <li>Manage stress and get adequate sleep.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Risk meter
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Risk Meter</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="meter-label"><span>Low Risk</span><span>High Risk</span></div>
            <div class="meter-track">
                <div class="meter-marker" style="left:{risk_pct}%;"></div>
            </div>
            <div class="meter-pct">{risk_pct}%</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📋 Edit Patient Details"):
                st.session_state.page = "details"
                st.rerun()
        with c2:
            if st.button("🔄 New Prediction"):
                st.session_state.patient = {}
                st.session_state.result = None
                st.session_state.page = "details"
                st.rerun()

        st.markdown(
            '<div class="disclaimer">⚕️ This is an automated estimate, not a medical '
            "diagnosis. Please consult a qualified healthcare professional for an "
            "accurate assessment.</div>",
            unsafe_allow_html=True,
        )

    render_footer()