import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Title ---
st.title("📊 Telecom Customer Churn Analysis & Prediction")

st.write("""
This dashboard analyzes customer churn behavior and predicts the risk of a customer leaving.
It helps businesses take proactive decisions to reduce customer loss.
""")

# --- Load Data ---
df = pd.read_csv("processed_data.csv")
model = pickle.load(open("model.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

# --- Model Performance ---
st.subheader("📌 Model Performance")
st.metric("Model Accuracy", f"{round(accuracy * 100, 2)}%")

# --- Key Metrics ---
st.subheader("📊 Key Insights")

total_customers = len(df)
churned = df["Churn"].sum()
churn_rate = (churned / total_customers) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Churn Rate", f"{churn_rate:.2f}%")

# --- Minimal Label Columns (only what we need) ---
df["Churn_label"] = df["Churn"].map({0: "Stayed", 1: "Churned"})

# --- IMPORTANT GRAPH 1 ---
st.subheader("📉 Overall Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Churn_label", ax=ax)
ax.set_title("Stayed vs Churned Customers")
st.pyplot(fig)

# --- IMPORTANT GRAPH 2 ---
st.subheader("💰 Monthly Charges Impact")

fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x="Churn_label", y="MonthlyCharges", ax=ax2)
ax2.set_title("Higher Charges → Higher Churn Risk")
st.pyplot(fig2)

# --- IMPORTANT GRAPH 3 ---
st.subheader("⏳ Customer Tenure Impact")

fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x="Churn_label", y="tenure", ax=ax3)
ax3.set_title("Lower Tenure → Higher Churn Risk")
st.pyplot(fig3)

# --- Sidebar ---
st.sidebar.header("🔍 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)

# --- Mapping ---
gender_map = {"Female": 0, "Male": 1}
yesno_map = {"No": 0, "Yes": 1}

# --- Input Data ---
input_df = df.drop("Churn", axis=1).iloc[0:1].copy()
input_df = input_df[[col for col in input_df.columns if "_label" not in col]]

input_df["gender"] = gender_map[gender]
input_df["SeniorCitizen"] = yesno_map[SeniorCitizen]
input_df["Partner"] = yesno_map[Partner]
input_df["Dependents"] = yesno_map[Dependents]
input_df["tenure"] = tenure
input_df["MonthlyCharges"] = MonthlyCharges
input_df["TotalCharges"] = TotalCharges

# --- Prediction ---
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("🔮 Churn Prediction")

# Result
if prediction == 1:
    st.error("⚠️ Customer likely to churn")
else:
    st.success("✅ Customer likely to stay")

# Probability
st.write(f"Churn Probability: {round(probability * 100, 2)}%")
st.progress(int(probability * 100))

# Risk Level
if probability < 0.3:
    st.success("🟢 Low Risk")
elif probability < 0.6:
    st.warning("🟠 Medium Risk")
else:
    st.error("🔴 High Risk")

# --- Business Recommendation ---
st.subheader("💡 Recommendation")

if probability > 0.6:
    st.write("👉 Offer discounts or loyalty plans to retain this customer.")
elif probability > 0.3:
    st.write("👉 Improve engagement and customer experience.")
else:
    st.write("👉 Customer is stable. Maintain service quality.")
