import streamlit as st
import pandas as pd
from predict_utils import load_models, predict_cardiac_disease

st.set_page_config(page_title="Medical | Heart App")
st.title("🩺 Medical Questionnaire")

# Load models and features
model1, model2, features1, features2 = load_models()

# Define questions for medical input
medical_inputs = {}
for feature in features1:
    medical_inputs[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", value=0.0, step=1.0, key=f"med_{feature}")

# Check if lifestyle data is already collected
if "lifestyle_data" not in st.session_state:
    st.warning("Please fill in the Lifestyle Questionnaire first.")
else:
    if st.button("🔍 Predict Heart Disease"):
        medical_df = pd.DataFrame([medical_inputs])[features1]
        lifestyle_df = pd.DataFrame([st.session_state["lifestyle_data"]])[features2]

        # Combine and predict
        input_values = list(medical_df.iloc[0].values) + list(lifestyle_df.iloc[0].values)
        result = predict_cardiac_disease(input_values, model1, model2, features1, features2)

        if result["Final Prediction"] == "Heart Disease Detected":
            st.success("💔 Heart Disease Detected")
        else:
            st.success("❤️ No Heart Disease Detected")

        with st.expander("🔬 Probability Scores"):
            st.write(f"**Medical Model Confidence:** {result['Medical Model Confidence']:.2%}")
            st.write(f"**Lifestyle Model Confidence:** {result['Lifestyle Model Confidence']:.2%}")
