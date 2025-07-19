import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Title and Description
st.set_page_config(page_title="AI Heart Failure Detection", layout="wide")
st.title("🫀 AI Heart Failure Detection System")

st.markdown("""
Welcome to the AI-powered heart failure risk prediction system. This tool uses machine learning to assess the risk of heart failure based on patient data.

---
""")

# Loading Data

data = pd.read_csv("heart_failure_prediction.csv")

# Show Data

if st.checkbox("🔍 Show Raw Dataset"):
    st.dataframe(data.head())

# Data Preprocessing

st.subheader("📊 Data Preprocessing & Overview")

st.write("### Summary Statistics")
st.write(data.describe())

# Encode categorical variables

df = data.copy()
df = pd.get_dummies(df, drop_first=True)

# Feature/Target Split

X = df.drop("Heart_Failure", axis=1)
y = df["Heart_Failure"]

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation

st.subheader("✅ Model Evaluation")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred)*100:.2f}%")
st.write(f"**ROC-AUC Score:** {roc_auc_score(y_test, y_prob)*100:.2f}%")

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Plot

st.write("### Confusion Matrix")
plt.figure(figsize=(8, 6))
fig1, ax1 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Reds", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# Feature Importance

st.write("### 🔍 Top 10 Important Features")
plt.figure(figsize=(8, 6))
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x=importance[:10], y=importance.index[:10], palette='coolwarm', ax=ax2)
st.pyplot(fig2)

# User Prediction Input

st.subheader("🧪 Predict Risk of Heart Failure")

# Getting user data

def get_user_input():
    user_data = {}
    for col in data.columns:
        if col == "Heart_Failure":
            continue  # Skip target column
        if data[col].dtype == "object" or data[col].nunique() < 10:
            # Assume categorical if object or few unique values
            options = sorted(data[col].dropna().unique())
            user_data[col] = st.selectbox(f"{col}", options)
        else:
            mean_val = float(data[col].mean())
            user_data[col] = st.number_input(f"{col}", value=mean_val)
    return pd.DataFrame([user_data])


user_input_df = get_user_input()
user_input_encoded = pd.get_dummies(user_input_df)
user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

if st.button("Predict Heart Failure Risk"):
    prediction = model.predict(user_input_encoded)[0]
    probability = model.predict_proba(user_input_encoded)[0][1]

    if prediction == 1:
        st.error(f"🚨 Prediction: **At Risk**\n\nLikelihood: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ Prediction: **Not at Risk**\n\nLikelihood: **{probability*100:.2f}%**")


