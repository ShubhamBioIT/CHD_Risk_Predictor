import streamlit as st

# Page config
st.set_page_config(
    page_title="CardioGuard AI - Advanced CHD Risk Assessment",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fpdf import FPDF
import io
from io import BytesIO
import time
from datetime import datetime, timedelta
import base64

# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load("Tuned_random_forest_model.pkl")
    stack_model = joblib.load("Stacking_classifier_model.pkl")
    return rf_model, stack_model

rf_model, stack_model = load_models()

# Load Lottie animation
@st.cache_data
def load_lottie(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

lottie_heart = load_lottie("heart.json")

# Advanced CSS Styling for Professional CHD Risk Dashboard Theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');

    /* Global styles */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif !important;
        background: #0d1828 !important; /* Deep blue-black for medical/professional look */
        color: #f7fafd !important; /* Very light blue for high contrast */
    }

    /* Hide default streamlit elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom background with animated gradient */
    .stApp {
        background: #0d1828 !important;
        background-image: linear-gradient(120deg, #16213e 0%, #0d1828 100%);
        background-attachment: fixed;
        animation: bgmove 12s ease-in-out infinite alternate;
    }
    @keyframes bgmove {
        0% {background-position: 0% 50%;}
        100% {background-position: 100% 50%;}
    }

    /* Main container styling */
    .main-container {
        background: rgba(13, 24, 40, 0.98);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Animated title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b6b, #ee5253, #54a0ff, #2ed573, #ffa502, #ff3838);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        animation: gradientShift 5s ease-in-out infinite;
        text-shadow: 0 4px 24px rgba(0,0,0,0.6);
        letter-spacing: 2px;
        filter: drop-shadow(0 2px 8px #000);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.3rem;
        color: #b8c6e5;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        text-shadow: 0 2px 8px #000;
        letter-spacing: 1px;
        animation: fadeIn 1.2s;
    }

    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #16213e 0%, #22304a 100%);
        color: #f7fafd;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.08);
        animation: fadeInUp 0.8s;
    }
    .info-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.85);
        border-color: #54a0ff;
    }

    /* Risk cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        background: rgba(13,24,40,0.95);
        box-shadow: 0 4px 16px rgba(0,0,0,0.7);
        color: #f7fafd;
        animation: fadeInUp 0.8s;
    }
    .risk-card:hover {
        transform: scale(1.03);
        border-color: #54a0ff;
    }
    .low-risk {
        background: linear-gradient(135deg, #2ed573 60%, #16213e 100%);
        color: #fff;
        border: 2px solid #2ed573;
        box-shadow: 0 0 16px #2ed57355;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #ffa502 60%, #16213e 100%);
        color: #fff;
        border: 2px solid #ffa502;
        box-shadow: 0 0 16px #ffa50255;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff3838 60%, #16213e 100%);
        color: #fff;
        border: 2px solid #ff3838;
        box-shadow: 0 0 16px #ff383855;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #54a0ff, #2ed573, #ff6b6b);
        color: #fff;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        outline: none;
        animation: pulse 2.5s infinite;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 25px #54a0ff55;
        background: linear-gradient(90deg, #2ed573, #54a0ff, #ff6b6b);
        color: #fff;
    }

    /* Input styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stSlider > div > div {
        background: rgba(22, 33, 62, 0.98) !important;
        border-radius: 10px !important;
        border: 2px solid #16213e !important;
        color: #f5f6fa !important;
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover,
    .stSlider > div > div:hover {
        border-color: #54a0ff !important;
        box-shadow: 0 5px 15px #54a0ff33;
    }
    .stSlider > div > div {
        padding: 1rem;
    }

    /* Metrics styling */
    .stMetric {
        background: rgba(22, 33, 62, 0.98) !important;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        border: 1px solid #16213e;
        color: #f5f6fa !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #16213e, #54a0ff);
        color: #fff !important;
        border-radius: 10px 10px 0 0;
        padding: 1rem;
        font-weight: 600;
        border-bottom: 1px solid #16213e;
    }
    .streamlit-expanderContent {
        background: rgba(22, 33, 62, 0.98) !important;
        border-radius: 0 0 10px 10px;
        padding: 1rem;
        border: 1px solid #16213e;
        color: #fff !important;
    }

    /* Progress bar */
    .progress-container {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #54a0ff, #2ed573);
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Floating elements */
    .floating-card {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(22, 33, 62, 0.98);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
        z-index: 1000;
        border: 1px solid #16213e;
        color: #fff;
    }

    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.7s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .fadeInUp {
        animation: fadeInUp 0.8s;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .pulse {
        animation: pulse 2.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1);}
        50% { transform: scale(1.04);}
        100% { transform: scale(1);}
    }
    .glow {
        animation: glow 1.5s infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 8px #54a0ff55;}
        to { box-shadow: 0 0 24px #54a0ff;}
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem;
        }
        .subtitle {
            font-size: 1rem;
        }
        .floating-card {
            position: relative;
            top: 0;
            right: 0;
            margin: 1rem 0;
        }
        .main-container {
            padding: 1rem;
        }
    }

    /* Health recommendation cards */
    .health-recommendation {
        background: linear-gradient(135deg, #16213e, #54a0ff 80%);
        color: #fff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        transition: all 0.3s ease;
        border: 1px solid #16213e;
        animation: fadeInUp 0.8s;
    }
    .health-recommendation:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 40px #54a0ff55;
        border-color: #54a0ff;
    }
    .recommendation-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #fff;
        text-shadow: 0 2px 8px #000;
    }
    .recommendation-content {
        font-size: 1rem;
        line-height: 1.6;
        color: #c8d6e5;
    }

    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, #ff6b6b, #ee5253 80%);
        color: #fff;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        animation: pulse 3s infinite;
        text-shadow: 0 2px 8px #000;
    }

    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #2ed573 60%, #16213e 100%);
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 5px 15px #2ed57355;
        border: 1px solid #2ed573;
        animation: fadeIn 0.7s;
    }

    /* Warning message styling */
    .warning-message {
        background: linear-gradient(135deg, #ffa502 60%, #16213e 100%);
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 5px 15px #ffa50255;
        border: 1px solid #ffa502;
        animation: fadeIn 0.7s;
    }

    /* Error message styling */
    .error-message {
        background: linear-gradient(135deg, #ff3838 60%, #16213e 100%);
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 5px 15px #ff383855;
        border: 1px solid #ff3838;
        animation: fadeIn 0.7s;
    }

    /* Table styles for dark mode */
    table, th, td {
        background: #16213e !important;
        color: #f5f6fa !important;
        border: 1px solid #34495e !important;
    }
    th {
        background: #34495e !important;
        color: #54a0ff !important;
    }
    tr:nth-child(even) {
        background: #16213e !important;
    }
    tr:nth-child(odd) {
        background: #101624 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'risk_percentage' not in st.session_state:
    st.session_state.risk_percentage = 0
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = "Low"
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Helper functions
def get_risk_color(risk_percentage):
    if risk_percentage < 30:
        return "#2ed573"
    elif risk_percentage < 60:
        return "#ffa502"
    else:
        return "#ff3838"

def get_risk_level(risk_percentage):
    if risk_percentage < 30:
        return "Low"
    elif risk_percentage < 60:
        return "Moderate"
    else:
        return "High"

def create_risk_gauge(risk_percentage, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'family': 'Poppins'}},
        delta={'reference': 30, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_risk_color(risk_percentage), 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 213, 115, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 165, 2, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(255, 56, 56, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': 'Poppins'},
        height=300
    )
    
    return fig

def generate_personalized_recommendations(risk_percentage, user_data):
    recommendations = {
        "nutrition": [],
        "exercise": [],
        "lifestyle": [],
        "medical": [],
        "mental_health": []
    }
    
    # Risk-based recommendations
    if risk_percentage < 30:  # Low Risk
        recommendations["nutrition"] = [
            "🥗 Maintain Mediterranean diet with olive oil, nuts, and fish",
            "🍎 Include 5-7 servings of fruits and vegetables daily",
            "🥜 Add omega-3 rich foods like salmon, walnuts, and flaxseeds",
            "🧂 Keep sodium intake under 2300mg per day",
            "🫐 Include antioxidant-rich berries and dark leafy greens"
        ]
        
        recommendations["exercise"] = [
            "🏃‍♂️ Maintain 150 minutes of moderate exercise weekly",
            "💪 Include strength training 2-3 times per week",
            "🚶‍♀️ Take 8,000-10,000 steps daily",
            "🧘‍♀️ Practice yoga or stretching 3 times weekly",
            "🏊‍♂️ Try swimming or cycling for cardiovascular health"
        ]
        
        recommendations["lifestyle"] = [
            "😴 Maintain 7-9 hours of quality sleep",
            "🚭 Continue avoiding smoking and secondhand smoke",
            "🍷 Limit alcohol to 1 drink/day (women) or 2 drinks/day (men)",
            "💧 Stay hydrated with 8-10 glasses of water daily",
            "🧘‍♂️ Practice stress management techniques"
        ]
        
        recommendations["medical"] = [
            "🩺 Annual health checkups with lipid panel",
            "🩸 Monitor blood pressure monthly",
            "📊 Track BMI and waist circumference",
            "💉 Stay up-to-date with vaccinations",
            "🦷 Regular dental checkups (poor oral health linked to heart disease)"
        ]
        
        recommendations["mental_health"] = [
            "🧠 Practice mindfulness meditation 10-15 minutes daily",
            "👥 Maintain strong social connections",
            "📚 Engage in mentally stimulating activities",
            "🎯 Set and achieve personal goals",
            "😊 Practice gratitude journaling"
        ]
    
    elif risk_percentage < 60:  # Moderate Risk
        recommendations["nutrition"] = [
            "🥗 Adopt strict Mediterranean or DASH diet",
            "🍎 Increase fruits and vegetables to 7-9 servings daily",
            "🐟 Include fatty fish 3-4 times per week",
            "🥜 Add plant-based proteins like beans and lentils",
            "🧂 Reduce sodium to under 1500mg daily",
            "🚫 Eliminate processed and trans fats completely",
            "🌾 Choose whole grains over refined carbohydrates"
        ]
        
        recommendations["exercise"] = [
            "🏃‍♂️ Increase to 200-300 minutes of moderate exercise weekly",
            "💪 Strength training 3-4 times per week",
            "🚶‍♀️ Aim for 10,000+ steps daily",
            "🏊‍♂️ Include 2-3 cardio sessions weekly",
            "🧘‍♀️ Daily yoga or stretching routine",
            "⏰ Break up sitting time every 30 minutes"
        ]
        
        recommendations["lifestyle"] = [
            "😴 Prioritize 7-9 hours of quality sleep",
            "🚭 Smoking cessation programs if applicable",
            "🍷 Limit alcohol to 3-4 drinks per week maximum",
            "💧 Increase water intake to 10-12 glasses daily",
            "🧘‍♂️ Daily stress management practices",
            "📱 Limit screen time and blue light exposure"
        ]
        
        recommendations["medical"] = [
            "🩺 Bi-annual comprehensive health checkups",
            "🩸 Weekly blood pressure monitoring",
            "📊 Monthly weight and BMI tracking",
            "💊 Discuss preventive medications with doctor",
            "🏥 Consider cardiac calcium scoring",
            "🩹 Monitor for diabetes risk factors"
        ]
        
        recommendations["mental_health"] = [
            "🧠 Daily meditation or mindfulness practice",
            "👥 Build and maintain social support network",
            "😌 Consider counseling for stress management",
            "🎯 Set realistic health goals with professional guidance",
            "😊 Practice positive psychology techniques"
        ]
    
    else:  # High Risk
        recommendations["nutrition"] = [
            "🥗 Strict therapeutic diet (consult nutritionist)",
            "🍎 9+ servings of fruits and vegetables daily",
            "🐟 Fatty fish 4+ times per week",
            "🥜 Daily nuts and seeds (unsalted)",
            "🧂 Sodium restriction to 1000-1500mg daily",
            "🚫 Complete elimination of processed foods",
            "🌾 100% whole grain choices",
            "🥛 Consider plant-based milk alternatives",
            "☕ Limit caffeine to 1-2 cups daily"
        ]
        
        recommendations["exercise"] = [
            "🏃‍♂️ Supervised exercise program (300+ minutes weekly)",
            "💪 Resistance training 4-5 times per week",
            "🚶‍♀️ 12,000+ steps daily with activity tracking",
            "🏊‍♂️ Low-impact cardio 4-5 times weekly",
            "🧘‍♀️ Daily flexibility and mobility work",
            "⏰ Active breaks every 20-30 minutes",
            "🎯 Work with exercise physiologist"
        ]
        
        recommendations["lifestyle"] = [
            "😴 Optimize sleep hygiene (7-9 hours nightly)",
            "🚭 Immediate smoking cessation with medical support",
            "🍷 Eliminate or severely limit alcohol",
            "💧 12+ glasses of water daily",
            "🧘‍♂️ Multiple daily stress reduction sessions",
            "📱 Digital detox periods",
            "🌡️ Monitor environmental stressors"
        ]
        
        recommendations["medical"] = [
            "🩺 Quarterly comprehensive health monitoring",
            "🩸 Daily blood pressure and heart rate monitoring",
            "📊 Weekly weight and symptom tracking",
            "💊 Medications as prescribed by cardiologist",
            "🏥 Regular cardiac imaging and stress tests",
            "🩹 Intensive diabetes and cholesterol management",
            "🚨 Emergency action plan for cardiac events"
        ]
        
        recommendations["mental_health"] = [
            "🧠 Professional stress management therapy",
            "👥 Cardiac rehabilitation support groups",
            "😌 Regular counseling sessions",
            "🎯 Professional goal setting and monitoring",
            "😊 Positive psychology interventions",
            "🧘‍♂️ Mindfulness-based stress reduction (MBSR)",
            "📞 24/7 mental health support access"
        ]
    
    # Personalized adjustments based on user data
    age = user_data.get('age', 50)
    sex = user_data.get('sex', 0)
    smoking = user_data.get('is_smoking', 0)
    diabetes = user_data.get('diabetes', 0)
    hypertension = user_data.get('prevalentHyp', 0)
    
    # Age-specific adjustments
    if age > 65:
        recommendations["exercise"].append("🦴 Include balance training to prevent falls")
        recommendations["medical"].append("🧠 Annual cognitive health screening")
        recommendations["nutrition"].append("🥛 Ensure adequate calcium and vitamin D")
    
    # Gender-specific adjustments
    if sex == 0:  # Female
        recommendations["medical"].append("🩺 Discuss hormone replacement therapy risks/benefits")
        recommendations["nutrition"].append("🌸 Include phytoestrogen-rich foods")
    
    # Condition-specific adjustments
    if smoking:
        recommendations["lifestyle"].insert(0, "🚭 URGENT: Smoking cessation is your #1 priority")
        recommendations["medical"].append("🫁 Pulmonary function testing")
    
    if diabetes:
        recommendations["nutrition"].append("🍯 Strict blood sugar management")
        recommendations["medical"].append("📊 HbA1c monitoring every 3 months")
    
    if hypertension:
        recommendations["nutrition"].append("🧂 Ultra-low sodium diet (<1500mg)")
        recommendations["medical"].append("🩸 Home blood pressure monitoring")
    
    return recommendations

def create_health_dashboard():
    st.markdown("### 📊 Your Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🫀 CHD Risk Score",
            value=f"{st.session_state.risk_percentage:.1f}%",
            delta=f"{st.session_state.risk_level} Risk"
        )
    
    with col2:
        st.metric(
            label="📈 Risk Category",
            value=st.session_state.risk_level,
            delta="Based on ML Analysis"
        )
    
    with col3:
        target_risk = max(0, st.session_state.risk_percentage - 10)
        st.metric(
            label="🎯 Target Risk",
            value=f"{target_risk:.1f}%",
            delta=f"-{st.session_state.risk_percentage - target_risk:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📅 Next Checkup",
            value="3 months",
            delta="Recommended"
        )

def create_interactive_risk_assessment():
    st.markdown("### 🔍 Interactive Risk Assessment")
    
    # Create risk factor radar chart
    categories = ['Age', 'Blood Pressure', 'Cholesterol', 'Smoking', 'Diabetes', 'BMI']
    
    # Normalize user values to 0-100 scale
    age_score = min(100, (st.session_state.user_data.get('age', 50) - 18) / 62 * 100)
    bp_score = min(100, (st.session_state.user_data.get('sysBP', 120) - 90) / 110 * 100)
    chol_score = min(100, (st.session_state.user_data.get('totChol', 200) - 100) / 300 * 100)
    smoke_score = st.session_state.user_data.get('is_smoking', 0) * 100
    diabetes_score = st.session_state.user_data.get('diabetes', 0) * 100
    bmi_score = min(100, (st.session_state.user_data.get('BMI', 25) - 18.5) / 21.5 * 100)
    
    values = [age_score, bp_score, chol_score, smoke_score, diabetes_score, bmi_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Risk Factors',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Add healthy baseline
    healthy_values = [30, 20, 30, 0, 0, 40]
    fig.add_trace(go.Scatterpolar(
        r=healthy_values,
        theta=categories,
        fill='toself',
        name='Healthy Baseline',
        line_color='rgb(46, 213, 115)',
        fillcolor='rgba(46, 213, 115, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Risk Factor Analysis",
        font=dict(family="Poppins"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_advanced_pdf_report(input_data, rf_prob, stack_prob, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(200, 15, "CardioGuard AI - Comprehensive CHD Risk Report", ln=True, align='C')
    
    # Date and time
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align='C')
    
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    
    risk_level = get_risk_level(stack_prob * 100)
    pdf.multi_cell(200, 8, f"Based on advanced machine learning analysis, your 10-year CHD risk is {stack_prob:.1%} ({risk_level} Risk). This report provides personalized recommendations for optimal cardiovascular health.")
    
    pdf.ln(5)
    
    # Risk Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Risk Analysis", ln=True)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(200, 8, f"Random Forest Model Prediction: {rf_prob:.2%}", ln=True)
    pdf.cell(200, 8, f"Stacking Ensemble Model Prediction: {stack_prob:.2%}", ln=True)
    pdf.cell(200, 8, f"Risk Classification: {risk_level}", ln=True)
    
    pdf.ln(5)
    
    # Patient Information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", '', 12)
    
    for key, value in input_data.items():
        pdf.cell(200, 6, f"{key}: {value}", ln=True)
    
    pdf.ln(5)
    
    # Key Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Key Recommendations", ln=True)
    pdf.set_font("Arial", '', 12)
    
    # Add top 3 recommendations from each category
    categories = ['nutrition', 'exercise', 'lifestyle', 'medical']
    for category in categories:
        if category in recommendations:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, f"{category.title()}:", ln=True)
            pdf.set_font("Arial", '', 10)
            for i, rec in enumerate(recommendations[category][:3]):
                safe_rec = rec.replace("•", "-").replace("🥗", "").replace("🍎", "").replace("🐟", "").replace("🥜", "").replace("🧂", "").replace("🚫", "").replace("🌾", "").replace("🥛", "").replace("☕", "").replace("🍳", "").replace("🥣", "").replace("🍞", "").replace("🥤", "").replace("🍲", "").replace("🍗", "").replace("🍝", "").replace("🥘", "").replace("🫐", "").replace("🍷", "").replace("💧", "").replace("🧘‍♂️", "").replace("🧠", "").replace("👥", "").replace("📚", "").replace("🎯", "").replace("😊", "").replace("🦴", "").replace("🩺", "").replace("🩸", "").replace("📊", "").replace("💉", "").replace("🦷", "").replace("🫁", "").replace("🍯", "").replace("🚭", "").replace("🩹", "").replace("🏥", "").replace("🚨", "").replace("📞", "").replace("😴", "").replace("📱", "").replace("🌡️", "").replace("😌", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🏃‍♂️", "").replace("🏃‍♀️", "").replace("🚶‍♂️", "").replace("🚶‍♀️", "").replace("🚴‍♂️", "").replace("🏊‍♀️", "").replace("🏋️‍♂️", "").replace("💪", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♀️", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "").replace("🧘‍♂️", "").replace("🧘‍♀️", "").replace("🤸‍♂️", "").replace("🤲", "")
                pdf.multi_cell(200, 6, f"- {safe_rec}")
            pdf.ln(2)
    
    # Generate PDF bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

def create_meal_plan_generator(risk_level):
    st.markdown("### 🍽️ Personalized Meal Plan Generator")
    
    meal_plans = {
        "Low": {
            "breakfast": [
                "🥣 Oatmeal with berries and walnuts",
                "🍳 Veggie omelet with whole grain toast",
                "🥤 Green smoothie with spinach and banana",
                "🥞 Whole grain pancakes with fresh fruit"
            ],
            "lunch": [
                "🥗 Mediterranean quinoa salad",
                "🐟 Grilled salmon with roasted vegetables",
                "🥪 Avocado and hummus wrap",
                "🍲 Lentil soup with whole grain bread"
            ],
            "dinner": [
                "🍗 Herb-crusted chicken with sweet potato",
                "🐟 Baked cod with steamed broccoli",
                "🍝 Whole grain pasta with marinara sauce",
                "🥘 Chickpea curry with brown rice"
            ],
            "snacks": [
                "🥜 Mixed nuts and seeds",
                "🍎 Apple with almond butter",
                "🥕 Carrot sticks with hummus",
                "🫐 Greek yogurt with berries"
            ]
        },
        "Moderate": {
            "breakfast": [
                "🥣 Steel-cut oats with flaxseeds",
                "🍳 Egg whites with spinach",
                "🥤 Protein smoothie with kale",
                "🍞 Ezekiel bread with avocado"
            ],
            "lunch": [
                "🥗 Kale Caesar salad with grilled chicken",
                "🐟 Wild salmon with quinoa",
                "🥪 Turkey and veggie lettuce wraps",
                "🍲 Vegetable bean soup"
            ],
            "dinner": [
                "🍗 Grilled chicken breast with asparagus",
                "🐟 Baked halibut with cauliflower rice",
                "🥘 Lentil and vegetable stew",
                "🍝 Zucchini noodles with turkey meatballs"
            ],
            "snacks": [
                "🥜 Almonds (10-15 pieces)",
                "🥒 Cucumber with tzatziki",
                "🍓 Berries with low-fat Greek yogurt",
                "🥕 Baby carrots with hummus"
            ]
        },
        "High": {
            "breakfast": [
                "🥣 Oat bran with fresh berries",
                "🍳 Egg white scramble with vegetables",
                "🥤 Green vegetable juice",
                "🍞 Whole grain toast with natural peanut butter"
            ],
            "lunch": [
                "🥗 Spinach salad with beans",
                "🐟 Steamed fish with brown rice",
                "🥪 Veggie-packed lettuce wraps",
                "🍲 Low-sodium vegetable soup"
            ],
            "dinner": [
                "🍗 Baked skinless chicken with herbs",
                "🐟 Grilled fish with steamed vegetables",
                "🥘 Bean and vegetable chili",
                "🍝 Whole grain pasta with vegetables"
            ],
            "snacks": [
                "🥜 Unsalted nuts (small portion)",
                "🍎 Fresh fruit",
                "🥕 Raw vegetables",
                "🫐 Low-fat yogurt"
            ]
        }
    }
    
    plan = meal_plans.get(risk_level, meal_plans["Low"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌅 Breakfast Options")
        for meal in plan["breakfast"]:
            st.markdown(f"• {meal}")
        
        st.markdown("#### 🥗 Lunch Options")
        for meal in plan["lunch"]:
            st.markdown(f"• {meal}")
    
    with col2:
        st.markdown("#### 🌙 Dinner Options")
        for meal in plan["dinner"]:
            st.markdown(f"• {meal}")
        
        st.markdown("#### 🍿 Healthy Snacks")
        for snack in plan["snacks"]:
            st.markdown(f"• {snack}")

def create_exercise_plan_generator(risk_level):
    st.markdown("### 🏃‍♂️ Personalized Exercise Plan")
    
    exercise_plans = {
        "Low": {
            "cardio": [
                "🚶‍♂️ Brisk walking 30 minutes, 5 days/week",
                "🏃‍♀️ Jogging 20 minutes, 3 days/week",
                "🚴‍♂️ Cycling 45 minutes, 2 days/week",
                "🏊‍♀️ Swimming 30 minutes, 2 days/week"
            ],
            "strength": [
                "💪 Full body workout 2-3 times/week",
                "🏋️‍♂️ Free weights 30 minutes sessions",
                "🤸‍♀️ Bodyweight exercises 3 times/week",
                "🧘‍♂️ Resistance band training"
            ],
            "flexibility": [
                "🧘‍♀️ Yoga 2-3 times/week",
                "🤸‍♂️ Dynamic stretching daily",
                "🧘‍♂️ Tai Chi once/week",
                "🤲 Foam rolling after workouts"
            ]
        },
        "Moderate": {
            "cardio": [
                "🚶‍♂️ Power walking 45 minutes, 5 days/week",
                "🏃‍♀️ Light jogging 25 minutes, 4 days/week",
                "🚴‍♂️ Stationary cycling 40 minutes, 3 days/week",
                "🏊‍♀️ Water aerobics 45 minutes, 2 days/week"
            ],
            "strength": [
                "💪 Supervised strength training 3 times/week",
                "🏋️‍♂️ Light weights with high reps",
                "🤸‍♀️ Functional movement exercises",
                "🧘‍♂️ Pilates 2 times/week"
            ],
            "flexibility": [
                "🧘‍♀️ Gentle yoga daily",
                "🤸‍♂️ Stretching routine 2 times/day",
                "🧘‍♂️ Meditation with movement",
                "🤲 Daily mobility work"
            ]
        },
        "High": {
            "cardio": [
                "🚶‍♂️ Supervised walking program daily",
                "🏃‍♀️ Cardiac rehabilitation exercises",
                "🚴‍♂️ Recumbent bike 20-30 minutes",
                "🏊‍♀️ Pool walking/light swimming"
            ],
            "strength": [
                "💪 Medical supervision required",
                "🏋️‍♂️ Light resistance training",
                "🤸‍♀️ Chair exercises if needed",
                "🧘‍♂️ Core strengthening"
            ],
            "flexibility": [
                "🧘‍♀️ Gentle stretching daily",
                "🤸‍♂️ Range of motion exercises",
                "🧘‍♂️ Breathing exercises",
                "🤲 Stress-reduction movement"
            ]
        }
    }
    
    plan = exercise_plans.get(risk_level, exercise_plans["Low"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🫀 Cardiovascular")
        for exercise in plan["cardio"]:
            st.markdown(f"• {exercise}")
    
    with col2:
        st.markdown("#### 💪 Strength Training")
        for exercise in plan["strength"]:
            st.markdown(f"• {exercise}")
    
    with col3:
        st.markdown("#### 🧘‍♀️ Flexibility")
        for exercise in plan["flexibility"]:
            st.markdown(f"• {exercise}")

def create_progress_tracker():
    st.markdown("### 📝 Personalized Health Action Checklist")

    st.markdown("""
    Use this checklist to track your progress on key heart health actions. Mark each item as you complete it and revisit regularly to stay on track!
    """)

    # Define checklist items based on risk level
    risk_level = st.session_state.get("risk_level", "Low")
    checklist = []

    if risk_level == "Low":
        checklist = [
            "Maintain a Mediterranean-style diet",
            "Exercise at least 150 minutes per week",
            "Monitor blood pressure monthly",
            "Get annual health checkups",
            "Practice stress management (e.g., meditation, yoga)",
            "Avoid smoking and limit alcohol",
            "Track your daily steps (aim for 8,000+)",
            "Get 7-9 hours of sleep nightly"
        ]
    elif risk_level == "Moderate":
        checklist = [
            "Adopt a DASH or Mediterranean diet strictly",
            "Increase exercise to 200+ minutes per week",
            "Monitor blood pressure weekly",
            "Schedule bi-annual health checkups",
            "Reduce sodium and processed foods",
            "Join a support group or health community",
            "Track weight and BMI monthly",
            "Limit alcohol to 3-4 drinks/week",
            "Practice daily stress reduction"
        ]
    else:  # High risk
        checklist = [
            "Consult a cardiologist for a personalized care plan",
            "Follow a therapeutic diet (consult a nutritionist)",
            "Participate in supervised exercise or cardiac rehab",
            "Monitor blood pressure and glucose daily",
            "Take prescribed medications regularly",
            "Schedule quarterly health checkups",
            "Eliminate smoking and alcohol completely",
            "Track symptoms and weight weekly",
            "Engage in professional stress management or counseling"
        ]

    # Use session state to persist checklist
    if "progress_checklist" not in st.session_state or len(st.session_state.progress_checklist) != len(checklist):
        st.session_state.progress_checklist = [False] * len(checklist)

    # Display checklist with checkboxes
    for i, item in enumerate(checklist):
        checked = st.checkbox(item, value=st.session_state.progress_checklist[i], key=f"check_{i}")
        st.session_state.progress_checklist[i] = checked


    # Show completion progress
    completed = sum(st.session_state.progress_checklist)
    total = len(checklist)
    st.progress(completed / total if total else 0)
    st.markdown(f"**{completed} of {total} actions completed**")

    # Motivational message
    if completed == total and total > 0:
        st.success("🎉 Congratulations! You've completed all recommended actions for your heart health. Keep up the great work!")
    elif completed > 0:
        st.info("👍 Great progress! Keep working through your checklist for optimal results.")
    else:
        st.warning("Let's get started! Begin by checking off your first action.")

def main():
    # Header with animation
    st.markdown('<div class="main-title">🩺 CardioGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Cardiovascular Risk Assessment & Personalized Healthcare Platform</div>', unsafe_allow_html=True)
    
    # Display Lottie animation if available
    if lottie_heart:
        st_lottie(lottie_heart, height=200, key="heart_animation")
    
    # Navigation tabs
    tab_labels = [
        "🏥 Risk Assessment",
        "📊 Health Dashboard",
        "💊 Personalized Care",
        "📈 Progress Tracking",
        "📚 Health Education",
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)
    
    with tab1:
        st.markdown("### 🧬 Comprehensive Health Assessment")
        
        # Reference table in expandable section
        with st.expander("📘 Reference Values for Healthy Individuals"):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                <h3 style="color: white; margin-bottom: 1rem;">🧾 Age-Based Health Reference Table</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: rgba(255,255,255,0.2);">
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">Age Group</th>
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">Systolic BP</th>
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">Diastolic BP</th>
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">Total Cholesterol</th>
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">BMI</th>
                                <th style="padding: 12px; border: 1px solid rgba(255,255,255,0.3);">Glucose</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">18–29</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">100–120</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">60–80</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">125–200</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">18.5–24.9</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–99</td></tr>
                            <tr><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">30–39</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">105–125</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">65–85</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">130–210</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">18.5–24.9</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–99</td></tr>
                            <tr><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">40–49</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">110–130</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–85</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">140–220</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">18.5–25.0</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–99</td></tr>
                            <tr><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">50–59</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">115–135</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–90</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">150–230</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">18.5–25.0</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–99</td></tr>
                            <tr><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">60+</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">120–140</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–90</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">160–240</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">19–26</td><td style="padding: 10px; border: 1px solid rgba(255,255,255,0.2);">70–105</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # User input form
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 100, 50, help="Your current age in years")
            sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex")
            is_smoking = st.selectbox("Smoking Status", ["No", "Yes"], help="Do you currently smoke?")
            cigsPerDay = st.slider("Cigarettes per Day", 0, 50, 0, help="Average number of cigarettes smoked daily")
            
        with col2:
            BPMeds = st.selectbox("Blood Pressure Medication", ["No", "Yes"], help="Are you taking BP medication?")
            prevalentStroke = st.selectbox("History of Stroke", ["No", "Yes"], help="Have you had a stroke?")
            prevalentHyp = st.selectbox("Hypertension", ["No", "Yes"], help="Diagnosed with high blood pressure?")
            diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Diagnosed with diabetes?")
        
        with col3:
            sysBP = st.slider("Systolic Blood Pressure", 90, 200, 120, help="Top number in BP reading")
            diaBP = st.slider("Diastolic Blood Pressure", 60, 140, 80, help="Bottom number in BP reading")
            totChol = st.slider("Total Cholesterol", 100, 400, 200, help="Total cholesterol level (mg/dL)")
            glucose = st.slider("Fasting Glucose", 50, 300, 100, help="Fasting blood glucose (mg/dL)")
            BMI = st.slider("Body Mass Index", 10.0, 50.0, 25.0, help="BMI calculation")
        
        # Real-time risk indicator
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            if sysBP > 140:
                st.markdown('<div class="error-message">⚠️ High Blood Pressure Alert</div>', unsafe_allow_html=True)
            elif sysBP > 130:
                st.markdown('<div class="warning-message">⚠️ Elevated Blood Pressure</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-message">✅ Normal Blood Pressure</div>', unsafe_allow_html=True)
        
        with col_risk2:
            if totChol > 240:
                st.markdown('<div class="error-message">⚠️ High Cholesterol Alert</div>', unsafe_allow_html=True)
            elif totChol > 200:
                st.markdown('<div class="warning-message">⚠️ Borderline High Cholesterol</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-message">✅ Normal Cholesterol</div>', unsafe_allow_html=True)
        
        with col_risk3:
            if BMI > 30:
                st.markdown('<div class="error-message">⚠️ Obesity Range</div>', unsafe_allow_html=True)
            elif BMI > 25:
                st.markdown('<div class="warning-message">⚠️ Overweight Range</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-message">✅ Normal Weight</div>', unsafe_allow_html=True)
        
        # Feature engineering (same as original)
        sex_encoded = 1 if sex == "Male" else 0
        is_smoking_encoded = 1 if is_smoking == "Yes" else 0
        BPMeds_encoded = 1 if BPMeds == "Yes" else 0
        prevalentStroke_encoded = 1 if prevalentStroke == "Yes" else 0
        prevalentHyp_encoded = 1 if prevalentHyp == "Yes" else 0
        diabetes_encoded = 1 if diabetes == "Yes" else 0
        
        def smoking_level_fn(cigs):
            if cigs == 0: return 0
            elif cigs <= 10: return 1
            elif cigs <= 20: return 2
            else: return 3
        
        def bmi_category_fn(bmi):
            if bmi < 18.5: return 0
            elif bmi < 25: return 1
            elif bmi < 30: return 2
            else: return 3
        
        bp_ratio = round(sysBP / diaBP, 2)
        chol_age_ratio = round(totChol / age, 2)
        smoking_level = smoking_level_fn(cigsPerDay)
        bmi_category = bmi_category_fn(BMI)
        
        # Prepare input data
        input_df = pd.DataFrame([[
            age, sex_encoded, is_smoking_encoded, BPMeds_encoded, prevalentStroke_encoded, 
            prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, glucose, 
            smoking_level, bp_ratio, chol_age_ratio, bmi_category
        ]], columns=[
            'age', 'sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
            'diabetes', 'totChol', 'sysBP', 'diaBP', 'glucose', 'smoking_level',
            'bp_ratio', 'chol_age_ratio', 'bmi_category'
        ])
        
        # Store user data in session state
        st.session_state.user_data = {
            'age': age, 'sex': sex_encoded, 'is_smoking': is_smoking_encoded,
            'BPMeds': BPMeds_encoded, 'prevalentStroke': prevalentStroke_encoded,
            'prevalentHyp': prevalentHyp_encoded, 'diabetes': diabetes_encoded,
            'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP, 'glucose': glucose,
            'BMI': BMI, 'cigsPerDay': cigsPerDay
        }
        
        # Prediction button
        if st.button("🩺 Analyze CHD Risk", help="Click to get your comprehensive risk assessment"):
            with st.spinner("🔄 Analyzing your data with advanced AI models..."):
                time.sleep(2)  # Simulate processing time
                
                # Make predictions
                rf_pred = rf_model.predict(input_df)[0]
                rf_proba = rf_model.predict_proba(input_df)[0][1]
                
                stack_pred = stack_model.predict(input_df)[0]
                stack_proba = stack_model.predict_proba(input_df)[0][1]
                
                # Update session state
                st.session_state.prediction_made = True
                st.session_state.risk_percentage = stack_proba * 100
                st.session_state.risk_level = get_risk_level(stack_proba * 100)
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Your CHD Risk Analysis Results")
                
                # Metrics display
                col4, col5 = st.columns(2)
                with col4:
                    st.metric(
                        "🤖 Random Forest Model", 
                        "CHD Risk" if rf_pred else "No Risk", 
                        delta=f"{rf_proba:.2%} probability"
                    )
                with col5:
                    st.metric(
                        "🧠 Stacking Ensemble Model", 
                        "CHD Risk" if stack_pred else "No Risk", 
                        delta=f"{stack_proba:.2%} probability"
                    )
                
                # Risk gauges
                col6, col7 = st.columns(2)
                with col6:
                    fig_rf = create_risk_gauge(rf_proba * 100, "Random Forest Risk Score")
                    fig_rf.update_layout(font={'color': "white", 'family': 'Poppins'})
                    fig_rf['layout']['paper_bgcolor'] = "rgba(0,0,0,0)"
                    fig_rf['layout']['plot_bgcolor'] = "rgba(0,0,0,0)"
                    fig_rf['layout']['title']['font']['color'] = "white"
                    st.plotly_chart(fig_rf, use_container_width=True)
                
                with col7:
                    fig_stack = create_risk_gauge(stack_proba * 100, "Stacking Model Risk Score")
                    fig_stack.update_layout(font={'color': "white", 'family': 'Poppins'})
                    fig_stack['layout']['paper_bgcolor'] = "rgba(0,0,0,0)"
                    fig_stack['layout']['plot_bgcolor'] = "rgba(0,0,0,0)"
                    fig_stack['layout']['title']['font']['color'] = "white"
                    st.plotly_chart(fig_stack, use_container_width=True)
                # Risk assessment message
                if stack_proba > 0.6:
                    st.markdown('<div class="error-message">🔴 HIGH RISK: Immediate medical consultation recommended</div>', unsafe_allow_html=True)
                elif stack_proba >= 0.3:
                    st.markdown('<div class="warning-message">🟡 MODERATE RISK: Lifestyle changes and monitoring advised</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-message">🟢 LOW RISK: Continue healthy lifestyle habits</div>', unsafe_allow_html=True)
                
                # Generate personalized recommendations
                recommendations = generate_personalized_recommendations(stack_proba * 100, st.session_state.user_data)
                
                # Generate and offer PDF download
                pdf_buffer = generate_advanced_pdf_report(
                    st.session_state.user_data, 
                    rf_proba, 
                    stack_proba, 
                    recommendations
                )
                
                st.download_button(
                    label="📄 Download Comprehensive Report",
                    data=pdf_buffer,
                    file_name=f"CardioGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Download your complete health assessment report"
                )
    
    with tab2:
        if st.session_state.prediction_made:
            create_health_dashboard()
            st.markdown("---")
            create_interactive_risk_assessment()
        else:
            st.markdown("### 📊 Complete Risk Assessment First")
            st.info("Please complete the risk assessment in the first tab to view your personalized dashboard.")
    
    with tab3:
        if st.session_state.prediction_made:
            st.markdown("### 💊 Your Personalized Healthcare Plan")
            
            # Get recommendations
            recommendations = generate_personalized_recommendations(
                st.session_state.risk_percentage, 
                st.session_state.user_data
            )
            
            # Display recommendations in organized sections
            rec_tabs = st.tabs(["🍽️ Nutrition", "🏃‍♂️ Exercise", "🧘‍♀️ Lifestyle", "🩺 Medical", "🧠 Mental Health"])
            
            with rec_tabs[0]:
                st.markdown("#### 🥗 Nutritional Recommendations")
                for rec in recommendations["nutrition"]:
                    st.markdown(f"• {rec}")
                
                st.markdown("---")
                create_meal_plan_generator(st.session_state.risk_level)
            
            with rec_tabs[1]:
                st.markdown("#### 🏋️‍♀️ Exercise Recommendations")
                for rec in recommendations["exercise"]:
                    st.markdown(f"• {rec}")
                
                st.markdown("---")
                create_exercise_plan_generator(st.session_state.risk_level)
            
            with rec_tabs[2]:
                st.markdown("#### 🧘‍♀️ Lifestyle Recommendations")
                for rec in recommendations["lifestyle"]:
                    st.markdown(f"• {rec}")
                st.markdown("---")
                st.markdown("#### 🧠 Mental Health Recommendations")
                for rec in recommendations["mental_health"]:
                    st.markdown(f"• {rec}")
            
            with rec_tabs[3]:
                st.markdown("#### 🩺 Medical Recommendations")
                for rec in recommendations["medical"]:
                    st.markdown(f"• {rec}")
            
            with rec_tabs[4]:
                st.markdown("#### 🧠 Mental Health Recommendations")
                for rec in recommendations["mental_health"]:
                    st.markdown(f"• {rec}")
        else:
            st.markdown("### 💊 Complete Risk Assessment First")
            st.info("Please complete the risk assessment in the first tab to view your personalized care plan.")
    
    with tab4:
        if st.session_state.prediction_made:
            create_progress_tracker()
        else:
            st.markdown("### 📈 Complete Risk Assessment First")
            st.info("Please complete the risk assessment in the first tab to view your progress dashboard.")
    
    with tab5:
        st.markdown("### 📚 Health Education & Resources")
        st.markdown("""
        - [American Heart Association - Prevention](https://www.heart.org/en/healthy-living)
        - [CDC Heart Disease Resources](https://www.cdc.gov/heartdisease/prevention.htm)
        - [WHO Cardiovascular Disease](https://www.who.int/health-topics/cardiovascular-diseases)
        - [NHS Heart Health](https://www.nhs.uk/live-well/healthy-body/heart-health/)
        - [Harvard Health - Heart Disease](https://www.health.harvard.edu/topics/heart-disease)
        """)
        st.markdown("---")
        st.markdown("#### ℹ️ Disclaimer")
        st.info("This tool is for educational purposes only and does not replace professional medical advice. Always consult your healthcare provider for personalized recommendations.")

if __name__ == "__main__":
    main()
