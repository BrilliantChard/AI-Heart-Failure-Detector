import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Background Setup ----------------

def set_background(theme):
    if theme == "Dark":
        overlay = "rgba(0, 0, 0, 0.6)"  
        text_color = "white"
    else:
        overlay = "rgba(255, 255, 255, 0.4)"  
        text_color = "black"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url("https://github.com/BrilliantChard/AI-Heart-Failure-Detector/raw/main/background.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .footer {{
            text-align: center;
            font-size: 16px;
            padding-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Setup Page ----------------

st.set_page_config(page_title="AI Heart Failure Detector", layout="wide")


# ---------------- Sidebar Menu ----------------

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Heart Failure Predictor App", "Data Analysis", "Contact"],
        icons=["house", "activity", "bar-chart", "envelope"],
        menu_icon="cast",
        default_index=0,
    )
    theme_choice = st.selectbox("🌓 Theme", ["Light", "Dark"])
set_background(theme_choice)


# ---------------- Load and Encode Data ----------------


@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_prediction.csv")

df = load_data()
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train model

X = df.drop("Heart_Failure", axis=1)
y = df["Heart_Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ---------------- Page: Home ----------------


if selected == "Home":
    st.markdown("<h1 style='text-align: center;'>💓 AI Heart Failure Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Empowering Early Detection with Machine Learning</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    Welcome to the **AI Heart Failure Detector App**!

    This tool allows you to predict the likelihood of **heart failure** based on clinical data using a trained AI model.

    ---

    ### 🧠 How to Use:
    1. Go to the **Heart Failure Predictor App** tab to make predictions.
    2. Choose between:
       - **Manual Input**: Fill out the form.
       - **Upload CSV File**: For batch predictions.
    3. Explore the **Data Analysis** tab to:
       - Visualize patient outcome distribution.
       - View feature correlation matrix.
       - Evaluate model performance.

    ---

    ⚠️ *Disclaimer: This tool is for educational purposes only and not for medical diagnosis.*
    """)


# ---------------- Page: Prediction ----------------


elif selected == "Heart Failure Predictor App":
    st.title("🧠 Heart Failure Risk Prediction")
    input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV File"])

    if input_method == "Manual Input":
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 1, 120, 60)
                gender = st.selectbox("Gender", ["Male", "Female"])
                chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])
                resting_bp = st.slider("Resting BP", 80, 200, 120)
                cholesterol = st.slider("Cholesterol", 100, 600, 220)

            with col2:
                fbs = st.radio("Fasting Blood Sugar > 120", [0, 1])
                ecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LVH"])
                max_hr = st.slider("Max Heart Rate", 60, 220, 150)
                angina = st.radio("Exercise Induced Angina", [0, 1])
                oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

            with col3:
                slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
                vessels = st.slider("Num Major Vessels", 0, 3, 0)
                thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
                bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
                physical = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

            diabetes = st.radio("Diabetes", [0, 1])
            smoking = st.radio("Smoking History", [0, 1])
            alcohol = st.radio("Alcohol Consumption", [0, 1])
            family = st.radio("Family History of Heart Failure", [0, 1])

            submit = st.form_submit_button("🔮 Predict")

        if submit:
            input_dict = {
                "Age": age,
                "Gender": label_encoders["Gender"].transform([gender])[0],
                "Chest_Pain_Type": label_encoders["Chest_Pain_Type"].transform([chest_pain])[0],
                "Resting_BP": resting_bp,
                "Cholesterol": cholesterol,
                "Fasting_Blood_Sugar": fbs,
                "Resting_ECG": label_encoders["Resting_ECG"].transform([ecg])[0],
                "Max_Heart_Rate": max_hr,
                "Exercise_Induced_Angina": angina,
                "Oldpeak": oldpeak,
                "Slope": label_encoders["Slope"].transform([slope])[0],
                "Num_Major_Vessels": vessels,
                "Thalassemia": label_encoders["Thalassemia"].transform([thal])[0],
                "Diabetes": diabetes,
                "Smoking_History": smoking,
                "Alcohol_Consumption": alcohol,
                "Physical_Activity_Level": label_encoders["Physical_Activity_Level"].transform([physical])[0],
                "Family_History": family,
                "BMI": bmi
            }
            input_df = pd.DataFrame([input_dict])
            prediction = model.predict(input_df)[0]
            result = "💔 Likely to Have Heart Failure" if prediction else "💚 Unlikely to Have Heart Failure"
            st.success(f"Prediction: **{result}**")

    elif input_method == "Upload CSV File":
        st.info("Upload a CSV file with the same structure as your dataset.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                for col in label_encoders:
                    if col in input_df.columns:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                preds = model.predict(input_df)
                input_df['Prediction'] = ["💔 Likely" if p == 1 else "💚 Unlikely" for p in preds]
                st.success("Prediction completed!")
                st.dataframe(input_df)

                csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------- Page: Data Analysis ----------------


elif selected == "Data Analysis":
    st.title("📊 Model Data Analysis")

    st.subheader("❤️ Heart Failure Count Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Heart_Failure", data=df, palette="Set2", ax=ax1)
    ax1.set_xticklabels(["No Failure", "Heart Failure"])
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("🧬 Feature Correlation Matrix")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df)

    st.subheader("🔍 Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu', ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)


# ---------------- Page: Contact ----------------


elif selected == "Contact":
    st.markdown("<h2 style='text-align: center;'>📬 Contact Me</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:16px;'>
        🔗 <a href='https://github.com/BrilliantChard' target='_blank'>GitHub</a><br>
        💼 <a href='https://www.linkedin.com/in/chard-odhiambo-57636136a/' target='_blank'>LinkedIn</a><br>
        📞 Phone: <a href="tel:+254797394105">+254797394105</a><br>
        📧 Email: <a href="mailto:chardodhiambo@gmail.com">chardodhiambo@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- Footer ----------------


st.markdown("""
<hr>
<div class='footer'>
    Made with ❤️ by <strong>Engineer Chard Odhiambo</strong><br>
    🔗 <a href='https://github.com/BrilliantChard' target='_blank'>GitHub</a> |
    💼 <a href='https://www.linkedin.com/in/chard-odhiambo-57636136a/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
