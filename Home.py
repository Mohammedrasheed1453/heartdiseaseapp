import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction App")
st.title("Heart Disease Prediction")

st.markdown("""
This app uses machine learning models trained on two datasets:
- **Medical Reports**
- **Lifestyle and Symptoms**
""")

if st.button("Start Prediction"):
    st.switch_page("pages/Lifestyle.py")
