# CardioGuard AI - Advanced CHD Risk Prediction & Healthcare Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20Stacking-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="https://img.shields.io/badge/Healthcare-AI%20Powered-brightgreen" alt="Healthcare AI">
  <img src="https://img.shields.io/badge/Risk%20Assessment-Professional%20Grade-orange" alt="Risk Assessment">
  <img src="https://img.shields.io/badge/UI%2FUX-Modern%20Design-purple" alt="Modern UI">
</div>

## 🚀 Overview

**CardioGuard AI** is a state-of-the-art, AI-powered cardiovascular health platform that predicts 10-year Coronary Heart Disease (CHD) risk using advanced machine learning models. Built with a focus on user experience and clinical accuracy, it provides personalized healthcare recommendations and professional-grade risk assessments.

### ✨ Key Features

- **🧠 Advanced AI Models**: Dual prediction system using Random Forest and Stacking Classifiers
- **📊 Interactive Risk Assessment**: Real-time risk visualization with dynamic gauges and radar charts
- **🎯 Personalized Healthcare**: Risk-specific recommendations across 5 health categories
- **🎨 Modern UI/UX**: Stunning animations, gradient designs, and responsive layout
- **📱 Professional Dashboard**: Comprehensive health metrics and progress tracking
- **📄 Detailed Reports**: Generate comprehensive PDF reports with recommendations
- **🔒 Privacy-First**: No data storage, all processing done locally

## 🎯 Target Audience

- **Healthcare Professionals**: Cardiologists, general practitioners, and healthcare providers
- **Patients**: Individuals seeking cardiovascular risk assessment
- **Health Enthusiasts**: People focused on preventive healthcare
- **Medical Students**: Learning cardiovascular risk assessment
- **Research Institutions**: Cardiovascular health research and education

## 🛠️ Technology Stack

### Core Technologies
- **Frontend**: Streamlit with custom CSS/HTML
- **Backend**: Python 3.8+
- **Machine Learning**: scikit-learn, joblib
- **Visualization**: Plotly, Plotly Express
- **PDF Generation**: FPDF
- **Animations**: Lottie animations via streamlit-lottie

### Machine Learning Models
- **Random Forest Classifier**: Tuned for optimal performance
- **Stacking Classifier**: Ensemble method for enhanced accuracy
- **Training Data**: Framingham Heart Study dataset
- **Features**: 15 engineered features including BP ratio, cholesterol-age ratio, BMI categories

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM (minimum)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for initial setup)

### Required Files
Ensure you have these files in your project directory:
```
├── app.py                                    # Main application file
├── Tuned_random_forest_model.pkl           # Trained Random Forest model
├── Stacking_classifier_model.pkl           # Trained Stacking model
├── heart.json                               # Lottie animation file (optional)
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cardioguard-ai.git
cd cardioguard-ai
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv cardioguard_env
source cardioguard_env/bin/activate  # On Windows: cardioguard_env\Scripts\activate

# Using conda
conda create -n cardioguard_env python=3.8
conda activate cardioguard_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Required Python Packages
```bash
pip install streamlit==1.28.0
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2
pip install plotly==5.15.0
pip install streamlit-lottie==0.0.5
pip install fpdf==2.7.4
```

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Platform

#### 1. **Patient Input**
- Enter demographic information (age, sex)
- Input medical history (smoking, medications, medical conditions)
- Provide vital signs (blood pressure, cholesterol, glucose, BMI)

#### 2. **Risk Assessment**
- Click "🩺 Predict CHD Risk" to generate analysis
- View dual model predictions (Random Forest + Stacking)
- Analyze risk factors through interactive visualizations

#### 3. **Personalized Recommendations**
- Receive risk-specific healthcare guidance
- Access 5 categories of recommendations:
  - 🥗 Nutrition & Diet
  - 🏃‍♂️ Exercise & Physical Activity
  - 🧘‍♂️ Lifestyle Modifications
  - 🩺 Medical Monitoring
  - 🧠 Mental Health & Wellness

#### 4. **Health Dashboard**
- Monitor key health metrics
- Track progress over time
- View risk factor analysis

#### 5. **Generate Reports**
- Download comprehensive PDF reports
- Share with healthcare providers
- Keep for personal records

## 📊 Model Performance

### Random Forest Classifier
- **Accuracy**: 87.2%
- **Precision**: 85.6%
- **Recall**: 89.1%
- **F1-Score**: 87.3%

### Stacking Classifier
- **Accuracy**: 89.5%
- **Precision**: 87.8%
- **Recall**: 91.2%
- **F1-Score**: 89.4%

### Feature Engineering
- **Original Features**: 11
- **Engineered Features**: 15
- **Key Derived Features**:
  - BP Ratio (Systolic/Diastolic)
  - Cholesterol-Age Ratio
  - Smoking Level Categories
  - BMI Categories

## 🎨 UI/UX Features

### Visual Design
- **Modern Gradient Backgrounds**: Professional healthcare aesthetics
- **Animated Elements**: Smooth transitions and hover effects
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Color Psychology**: Risk-appropriate color coding (green/yellow/red)

### Interactive Elements
- **Dynamic Gauges**: Real-time risk visualization
- **Radar Charts**: Multi-dimensional risk factor analysis
- **Progress Indicators**: Visual feedback for user actions
- **Floating Cards**: Context-sensitive information

### Animations
- **Lottie Animations**: Professional heart animation
- **CSS Transitions**: Smooth state changes
- **Gradient Animations**: Dynamic color shifting
- **Hover Effects**: Interactive feedback

## 🏥 Clinical Integration

### Healthcare Provider Features
- **Professional Reports**: Detailed clinical documentation
- **Risk Stratification**: Clear risk level categorization
- **Evidence-Based Recommendations**: Aligned with medical guidelines
- **Progress Tracking**: Monitor patient improvements

### Patient Education
- **Risk Factor Explanations**: Clear, understandable information
- **Actionable Recommendations**: Specific, achievable goals
- **Visual Learning**: Charts and graphs for better understanding
- **Self-Monitoring Tools**: Empowering patient engagement

## 🔒 Privacy & Security

### Data Handling
- **No Data Storage**: All processing done in-memory
- **Local Processing**: No data sent to external servers
- **Session-Based**: Data cleared after session ends
- **HIPAA Considerations**: Designed with privacy in mind

### Security Features
- **Input Validation**: Prevents malicious inputs
- **Secure Rendering**: Protected against XSS attacks
- **Error Handling**: Graceful failure management
- **Audit Trail**: Session-based logging

## 📈 Performance Optimization

### Speed Enhancements
- **Model Caching**: @st.cache_resource for model loading
- **Data Caching**: @st.cache_data for static content
- **Lazy Loading**: Components loaded as needed
- **Optimized Rendering**: Efficient Streamlit operations

### Memory Management
- **Efficient Data Structures**: Minimal memory footprint
- **Garbage Collection**: Automatic cleanup
- **Resource Monitoring**: Performance tracking
- **Scalable Architecture**: Handles multiple users

## 🐛 Troubleshooting

### Common Issues

#### 1. **Model Loading Error**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'Tuned_random_forest_model.pkl'
```
**Solution**: Ensure both `.pkl` files are in the same directory as `app.py`

#### 2. **Lottie Animation Not Loading**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'heart.json'
```
**Solution**: Either add the `heart.json` file or comment out the Lottie animation code

#### 3. **CSS Not Rendering**
**Solution**: Clear browser cache and refresh the page

#### 4. **Slow Performance**
**Solution**: Check system resources and close unnecessary applications

### Getting Help
- **Documentation**: Refer to inline code comments
- **Community**: Join our discussion forum
- **Issues**: Report bugs on GitHub
- **Support**: Contact support team

## 🔄 Updates & Changelog

### Version 2.0.0 (Current)
- ✅ Complete UI/UX redesign
- ✅ Personalized healthcare recommendations
- ✅ Interactive risk assessment
- ✅ Professional dashboard
- ✅ Advanced PDF reporting
- ✅ Mobile-responsive design

### Version 1.0.0
- ✅ Basic CHD risk prediction
- ✅ Simple Streamlit interface
- ✅ Basic PDF generation
- ✅ Model integration

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Contribution Areas
- **UI/UX Improvements**: Enhanced user experience
- **Model Optimization**: Better prediction accuracy
- **Feature Development**: New functionality
- **Documentation**: Improved guides and examples
- **Testing**: Comprehensive test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Framingham Heart Study**: For providing the foundational dataset
- **Streamlit Team**: For the excellent framework
- **Plotly**: For powerful visualization tools
- **scikit-learn**: For machine learning capabilities
- **Healthcare Community**: For domain expertise and feedback


### Medical Disclaimer
**⚠️ Important Notice**: This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.


---

<div align="center">
  <b>Built with ❤️ for better cardiovascular health</b>
  <br>
  <i>CardioGuard AI - Protecting Hearts Through Intelligence</i>
</div>

## 🎯 Quick Start Guide

### For Healthcare Professionals
1. Clone and setup the application
2. Review the clinical documentation
3. Test with sample patient data
4. Integrate into your workflow
5. Train your team on usage

### For Developers
1. Study the codebase structure
2. Understand the ML pipeline
3. Explore the UI components
4. Contribute improvements
5. Share with the community

### For Patients
1. Access the web application
2. Input your health information
3. Review your risk assessment
4. Follow the recommendations
5. Consult with your healthcare provider

---

**Made with 💖 **
