# File: pages/2_Data_Analysis.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- SETUP ----------------
st.set_page_config(page_title="Data Analysis", layout="wide")
st.title("📊 Model Data Analysis")

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

# ---------------- MODEL TRAINING ----------------
X = df.drop("Heart_Failure", axis=1)
y = df["Heart_Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- HEART FAILURE COUNT ----------------
st.subheader("❤️ Heart Failure Count Distribution")
st.write("Distribution of patients with and without heart failure in the training dataset.")
fig1, ax1 = plt.subplots()
sns.countplot(x="Heart_Failure", data=df, palette="Set2", ax=ax1)
ax1.set_xticklabels(["No Failure", "Heart Failure"])
ax1.set_ylabel("Count")
st.pyplot(fig1)

# ---------------- CORRELATION MATRIX ----------------
st.subheader("🧬 Feature Correlation Matrix")
st.write("This matrix shows how input features correlate with each other.")
corr = df.corr()
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# ---------------- CLASSIFICATION REPORT ----------------
st.subheader("📈 Classification Report")
st.write("Performance metrics of the Random Forest model.")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df)

# ---------------- CONFUSION MATRIX ----------------
st.subheader("🔍 Confusion Matrix")
fig3, ax3 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu', ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

st.success("End of analysis. You can explore more by testing different models or features.")
