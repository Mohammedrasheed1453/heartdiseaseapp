import streamlit as st
import pandas as pd

st.set_page_config(page_title="Lifestyle | Heart App")
st.title("🏃 Lifestyle & Symptoms Questionnaire")

# Load dataset to get columns
df = pd.read_csv("Dataset_2.csv")
lifestyle_features = df.drop(columns=["Heart_Risk"]).columns.tolist()

# Define readable labels (fallback to feature name if not mapped)
question_labels = {
    "Smoking": "Do you smoke?",
    "Obesity": "Are you obese?",
    "Family_History": "Family history of heart disease?",
    "High_BP": "Do you have high blood pressure?",
    "High_Cholesterol": "Do you have high cholesterol?",
    "Diabetes": "Are you diabetic?",
    "Sedentary_Lifestyle": "Sedentary lifestyle?",
    "Chronic_Stress": "Do you experience chronic stress?",
    "Shortness_of_Breath": "Do you have shortness of breath?",
    "Fatigue": "Are you frequently fatigued?",
    "Palpitations": "Do you feel palpitations?",
    "Dizziness": "Do you feel dizzy?",
    "Swelling": "Do you experience swelling in legs/feet?",
    "Pain_Arms_Jaw_Back": "Pain in arms, jaw or back?",
    "Cold_Sweats_Nausea": "Cold sweats or nausea?",
    "Chest_Pain": "Do you experience chest pain?",
    "Gender": "What is your gender? (1 = Male, 0 = Female)",
    "Age": "What is your age?"
}

responses = {}
for feature in lifestyle_features:
    label = question_labels.get(feature, f"Enter value for {feature}")
    # Choose input type
    if feature.lower() == "age":
        responses[feature] = st.number_input(label, min_value=0, max_value=120, step=1, key=feature)
    else:
        responses[feature] = st.radio(label, [0, 1], horizontal=True, key=feature)

if st.button("✅ Continue to Medical Questions"):
    st.session_state["lifestyle_data"] = responses
    st.switch_page("pages/Medical.py")
