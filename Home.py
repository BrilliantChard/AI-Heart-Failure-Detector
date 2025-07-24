import streamlit as st
from PIL import Image

def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/BrilliantChard/AI-Heart-Failure-Detector/raw/main/background.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .top-logo img {{
            width: 100%;
            height: 100px;
            object-fit: cover;
        }}
        .footer {{
            text-align: center;
            font-size: 16px;
            padding-top: 20px;
        }}
        .sidebar-links {{
            font-size: 14px;
            padding: 10px;
            line-height: 1.6;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Page Setup
st.set_page_config(page_title="AI Heart Failure Detector", layout="wide")
set_background()


# Title and Subtitle
st.markdown("<h1 style='text-align: center;'>💓 AI Heart Failure Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Empowering Early Detection with Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Links
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class='sidebar-links'>
    🔗 <a href='https://github.com/BrilliantChard' target='_blank'>GitHub</a><br>
    💼 <a href='https://www.linkedin.com/in/chard-odhiambo/' target='_blank'>LinkedIn</a><br>
    📞 Phone: <a href="tel:+254797394105">+254797394105</a><br>
    📧 Email: <a href="mailto:chardodhiambo@gmail.com">chardodhiambo@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Introduction
st.markdown("""
Welcome to the **AI Heart Failure Detector App**!

This tool allows you to predict the likelihood of **heart failure** based on clinical data using a trained AI model.

---

### 🧠 How to Use:
1. Go to the **Heart Failure Prediction** tab via the sidebar to predict heart failure.
2. Choose your input method:
   - **Manual Input**: Fill out the form with patient data.
   - **Upload CSV File**: Upload a CSV file for bulk predictions.
3. Visit the **Data Analysis** tab to:
   - Visualize heart failure count distribution.
   - Examine feature correlation matrix.
   - Evaluate model performance (accuracy, precision, recall, F1-score).

---

🗂 **Navigation**: Use the sidebar to switch between:
- 🏠 Home
- 📊 Heart Failure Prediction App
- 📈 Data Analysis

⚠️ *Disclaimer: This tool is for educational purposes only and not for medical diagnosis.*
""")

# Footer with GitHub and LinkedIn
st.markdown("""
<hr>
<div class='footer'>
    Made with ❤️ by <strong>Engineer Chard Odhiambo</strong><br>
    🔗 <a href='https://github.com/BrilliantChard' target='_blank'>GitHub</a> |
    💼 <a href='https://www.linkedin.com/in/chard-odhiambo/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
