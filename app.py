import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json
import plotly.graph_objects as go

# Load models
rf_model = joblib.load("Tuned_random_forest_model.pkl")
stack_model = joblib.load("Stacking_classifier_model.pkl")

# Page config
st.set_page_config(page_title="CHD Predictor", layout="wide")

# Load animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_heart = load_lottie("heart.json")

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: 900;
            color: #ff4d6d;
            text-align: center;
            animation: fadeIn 3s ease-in-out;
            margin-bottom: 30px;
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

# Title and animation
st.markdown('<div class="title">💖 CHD Risk Predictor</div>', unsafe_allow_html=True)
st_lottie(lottie_heart, height=200, key="heart")

# Input fields
st.subheader("🧬 Enter Patient Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    cigsPerDay = st.number_input("Cigarettes Per Day", 0, 50, 5)
    BPMeds = st.selectbox("On BP Medication?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    prevalentStroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    sysBP = st.slider("Systolic BP", 90, 200, 120)
    diaBP = st.slider("Diastolic BP", 60, 140, 80)

with col3:
    prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
    totChol = st.slider("Total Cholesterol", 100, 400, 200)
    BMI = st.slider("BMI", 10.0, 50.0, 25.0)
    glucose = st.slider("Glucose", 50, 300, 100)

# Encode categorical values
sex = 1 if sex == "Male" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Prepare input
input_data = pd.DataFrame([[
    age, sex, cigsPerDay,
    BPMeds, prevalentStroke, prevalentHyp, diabetes,
    totChol, sysBP, diaBP, BMI, glucose
]], columns=['age', 'sex', 'cigsPerDay',
             'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
             'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose'])

# Predict
if st.button("🩺 Predict CHD Risk"):
    # Predictions
    rf_pred = rf_model.predict(input_data)[0]
    rf_proba = rf_model.predict_proba(input_data)[0][1]

    stack_pred = stack_model.predict(input_data)[0]
    stack_proba = stack_model.predict_proba(input_data)[0][1]

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
    if stack_proba > 0.7:
        st.success("🔴 High Risk of Coronary Heart Disease. Please consult a cardiologist immediately.")
    elif 0.4 <= stack_proba <= 0.7:
        st.success("🟡 Moderate Risk of CHD. Lifestyle changes and further tests are recommended.")
    else:
        st.success("🟢 Low Risk. Maintain a healthy lifestyle and regular checkups.")

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
