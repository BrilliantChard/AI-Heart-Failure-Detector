import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="AI Heart Failure Detector", layout="wide")

# Title and Description
st.title("💓 AI Heart Failure Detector")
st.markdown("""
This app uses **Logistic Regression** to detect the risk of heart failure based on clinical features. 
Upload your own dataset or use the default Kaggle heart failure dataset.
""")

# Load Data
@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/ChardOmolo/data-repo/main/heart.csv"
    return pd.read_csv(url)

st.sidebar.header("🔍 Data Options")
upload_file = st.sidebar.file_uploader("Upload your heart failure data (.csv)", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.success("✅ Custom dataset loaded!")
else:
    df = load_default_data()
    st.info("📄 Using default Kaggle dataset")

# Display data
with st.expander("📊 Preview Data"):
    st.dataframe(df.head())

# Sidebar Info
st.sidebar.markdown("### Features Used")
st.sidebar.write(df.columns.tolist())

# Preprocessing
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Logistic Regression", "📈")
col2.metric("Accuracy", f"{acc*100:.2f}%", "✅")
col3.metric("Samples", f"{df.shape[0]}", "📄")

# Visualization - Confusion Matrix
st.subheader("📌 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# User Prediction
st.sidebar.subheader("🧪 Try Custom Prediction")
age = st.sidebar.slider("Age", 20, 100, 45)
chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 200)
bp = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 120)
maxhr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.5)
sex = st.sidebar.selectbox("Sex", options=[0, 1], help="0: Female, 1: Male")
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1])

if st.sidebar.button("Predict"):
    input_data = np.array([[age, sex, chol, bp, maxhr, oldpeak, fbs, exang]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.sidebar.success("✅ High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease")

# Footer
st.markdown("---")
st.markdown("Created by **Engineer Chard Omolo** | Data from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")
