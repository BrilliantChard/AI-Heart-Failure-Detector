import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- SETUP ----------------
st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
st.title("🧠 Heart Failure Risk Prediction")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_prediction.csv")

df = load_data()

# ---------------- ENCODING ----------------
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---------------- TRAIN MODEL ----------------
X = df.drop("Heart_Failure", axis=1)
y = df["Heart_Failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- INPUT METHOD ----------------
input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV File"])

# ---------------- MANUAL INPUT ----------------
if input_method == "Manual Input":
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=60)
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

    def encode_input():
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
        return pd.DataFrame([input_dict])

    if submit:
        input_data = encode_input()
        prediction = model.predict(input_data)[0]
        result = "💔 Likely to Have Heart Failure" if prediction else "💚 Unlikely to Have Heart Failure"
        st.success(f"Prediction: **{result}**")

# ---------------- CSV UPLOAD ----------------
elif input_method == "Upload CSV File":
    st.info("Upload a CSV file with the same structure as your dataset.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            for col in label_encoders:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])
            predictions = model.predict(input_df)
            input_df['Prediction'] = ["💔 Likely" if p == 1 else "💚 Unlikely" for p in predictions]
            st.success("Prediction completed!")
            st.dataframe(input_df)

            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing the file: {e}")
