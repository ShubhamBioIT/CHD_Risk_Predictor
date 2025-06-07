import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json
import plotly.graph_objects as go
from fpdf import FPDF
import io
from io import BytesIO

# Load models
rf_model = joblib.load("Tuned_random_forest_model.pkl")
stack_model = joblib.load("Stacking_classifier_model.pkl")

# Page config
st.set_page_config(page_title="CHD Predictor", layout="wide")

# Load Lottie animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_heart = load_lottie("heart.json")


def generate_pdf_report(input_data, prob, verdict_text, risk_level):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(255, 77, 109)
    pdf.cell(200, 10, "CHD Risk Prediction Report", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)

    pdf.cell(200, 10, "Input Details:", ln=True)
    for col, val in input_data.items():
        pdf.cell(200, 8, f"{col}: {val}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, f"CHD Risk Probability (Stacking Model): {prob:.2%}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, f"Final Verdict: {verdict_text}", ln=True)

    pdf.ln(5)
    if risk_level == "High":
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(200, 10, "You are at HIGH risk of Coronary Heart Disease. Immediate consultation with a cardiologist is strongly advised.")
    elif risk_level == "Moderate":
        pdf.set_text_color(255, 165, 0)
        pdf.multi_cell(200, 10, "You are at MODERATE risk. Consider lifestyle changes, follow-up tests, and medical guidance.")
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.multi_cell(200, 10, "You are at LOW risk. Keep up your healthy lifestyle and attend regular checkups.")

    # ✅ Generate PDF bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# --- Reset Page ---
if st.button("🔄 Reset"):
    st.rerun()

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: 900;
            color: #ff4d6d;
            text-align: center;
            animation: fadeIn 3s ease-in-out;
            margin-bottom: 40px;
        }
        .stButton > button {
            color: white !important;
            background-color: #ff4d6d !important;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: black !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">💖 CHD Risk Predictor</div>', unsafe_allow_html=True)
st_lottie(lottie_heart, height=200, key="heart")

# Expander Reference Table
with st.expander("📘 Show Reference Table For Healthy Individual (Based on Age)"):
    st.markdown("""
### 🧾 Reference Table for Healthy Individuals (Age-Wise)

| Age Group | Systolic BP | Diastolic BP | Total Cholesterol | BMI       | Fasting Glucose |
|-----------|-------------|---------------|--------------------|-----------|------------------|
| 18–29     | 100–120     | 60–80         | 125–200            | 18.5–24.9 | 70–99            |
| 30–39     | 105–125     | 65–85         | 130–210            | 18.5–24.9 | 70–99            |
| 40–49     | 110–130     | 70–85         | 140–220            | 18.5–25.0 | 70–99            |
| 50–59     | 115–135     | 70–90         | 150–230            | 18.5–25.0 | 70–99            |
| 60+       | 120–140     | 70–90         | 160–240            | 19–26     | 70–105           |
""")

# User Input
st.subheader("🧬 Enter Patient Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    is_smoking = st.selectbox("Do You Smoke?", ["No", "Yes"])
    cigsPerDay = st.slider("Cigarettes per Day", 0, 50, 0)
    
with col2:
    BPMeds = st.selectbox("On BP Medication?", ["No", "Yes"])
    prevalentStroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

with col3:
    sysBP = st.slider("Systolic BP", 90, 200, 120)
    diaBP = st.slider("Diastolic BP", 60, 140, 80)
    totChol = st.slider("Total Cholesterol", 100, 400, 200)
    glucose = st.slider("Glucose", 50, 300, 100)
    BMI = st.slider("BMI (for category)", 10.0, 50.0, 25.0)

# Encoding
sex = 1 if sex == "Male" else 0
is_smoking = 1 if is_smoking == "Yes" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Feature Engineering
def smoking_level_fn(cigs):
    if cigs == 0:
        return 0
    elif cigs <= 10:
        return 1
    elif cigs <= 20:
        return 2
    else:
        return 3

def bmi_category_fn(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

bp_ratio = round(sysBP / diaBP, 2)
chol_age_ratio = round(totChol / age, 2)
smoking_level = smoking_level_fn(cigsPerDay)
bmi_category = bmi_category_fn(BMI)

# Final Input for Model
input_df = pd.DataFrame([[
    age, sex, is_smoking, BPMeds, prevalentStroke, prevalentHyp, diabetes,
    totChol, sysBP, diaBP, glucose, smoking_level, bp_ratio, chol_age_ratio, bmi_category
]], columns=[
    'age', 'sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
    'diabetes', 'totChol', 'sysBP', 'diaBP', 'glucose', 'smoking_level',
    'bp_ratio', 'chol_age_ratio', 'bmi_category'
])

# Predict
if st.button("🩺 Predict CHD Risk"):
    # Predictions
    rf_pred = rf_model.predict(input_df)[0]
    rf_proba = rf_model.predict_proba(input_df)[0][1]

    stack_pred = stack_model.predict(input_df)[0]
    stack_proba = stack_model.predict_proba(input_df)[0][1]

    # Show metrics
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Random Forest Prediction", 
                  "CHD Risk" if rf_pred else "No Risk", 
                  delta=f"{rf_proba:.2%}")
    with col5:
        st.metric("Stacked Model Prediction", 
                  "CHD Risk" if stack_pred else "No Risk", 
                  delta=f"{stack_proba:.2%} Probability")

    # Gauges
    col6, col7 = st.columns(2)
    for idx, (model_name, prob) in enumerate([("Random Forest", rf_proba), ("Stacking Classifier", stack_proba)]):
        with [col6, col7][idx]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4d6d"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "crimson"}
                    ],
                }
            ))
            fig.update_layout(
                title=f"{model_name} Risk Score",
                height=250,  # Increased height
                margin=dict(t=50, b=10, l=10, r=10),
                )

            st.plotly_chart(fig, use_container_width=True)

    # Final verdict based on stacking model
    if stack_proba > 0.6:
        st.success("🔴 High Risk of Coronary Heart Disease. Please consult a cardiologist immediately.")
    elif 0.3 <= stack_proba <= 0.6:
        st.success("🟡 Moderate Risk of CHD. Lifestyle changes and further tests are recommended.")
    else:
        st.success("🟢 Low Risk. Maintain a healthy lifestyle and regular checkups.")

    # Generate and download PDF
   
    input_summary = input_df.iloc[0].to_dict()
    risk_label = "High" if stack_proba > 0.6 else "Moderate" if stack_proba >= 0.3 else "Low"
    verdict = "High Risk" if risk_label == "High" else "Moderate Risk" if risk_label == "Moderate" else "Low Risk"

    pdf_buffer = generate_pdf_report(input_df.iloc[0].to_dict(), stack_proba, verdict, risk_label)

    st.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer,
    file_name="CHD_Risk_Report.pdf",
    mime="application/pdf"
    )

    # Explanation Expander
    with st.expander("ℹ️ Understanding the CHD Risk Predictions (Important)"):
        st.markdown("""
### 🤔 Why Doesn’t the Model Predict 100% Risk for Extreme Values?

This tool uses **advanced machine learning models** like Random Forests and Stacking Classifiers, which are designed to predict the **probability** of developing Coronary Heart Disease (CHD) within 10 years.

#### Here's what you need to know:

---

#### 🧠 1. Probabilistic, Not Certain

These models **don't make binary or guaranteed predictions**. Instead, they provide a probability (e.g., 0.78 = 78%) that represents the **likelihood**, based on data — not a certainty.

Even a person with very high risk factors might not get CHD. So, the model learns from **real-world outcomes**, not just assumptions.

---

#### 📊 2. Real-World Based, Not Rule-Based

The model is trained on real patient data from the famous **Framingham Heart Study**. It has seen that:
- Many people with *bad* health markers **still didn't get CHD**,
- Some with *moderate markers* **did**.

So, your predicted risk reflects this **real-world uncertainty** — and that's what makes it reliable.

---

#### ⚖️ 3. Regularized to Avoid Overconfidence

To prevent incorrect or overly confident predictions, these models apply internal safeguards:
- **Random Forests** average decisions across trees,
- **Logistic Regression** applies regularization (penalties) to avoid extreme outputs.

So, even if all values are at max, a 100% prediction is **rare and not realistic** medically.

---

#### 💡 4. This Is Actually a Good Sign!

A model predicting 100% is often **overfitted**, meaning it's too confident and won't generalize well.

Here, getting **accurate and cautious probabilities like 70–80%** shows that the model is:
- Balanced ✅
- Well-calibrated ✅
- Not biased ✅
- Data-driven ✅

---

#### 🧬 5. What Makes This Tool Reliable?

- ✅ Trained on clean, balanced, real medical data
- ✅ Models fine-tuned for maximum precision and recall
- ✅ Threshold-tuned for best class separation
- ✅ Ensemble learning (Stacking) for robust prediction
- ✅ Calibrated probabilities based on clinical outcomes

---

### ✅ Final Takeaway

This tool gives you a **personalized, data-driven CHD probability** — not a yes/no label.

Use it as a **guiding health insight**, not a diagnostic verdict. 💖
    """)
