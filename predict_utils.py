# # predict_utils.py

# import joblib
# import pandas as pd

# def load_models():
#     model1 = joblib.load("model1_naive_bayes.pkl")
#     model2 = joblib.load("model2_gb.pkl")
#     features1 = joblib.load("model1_features.pkl")
#     features2 = joblib.load("model2_features.pkl")
#     return model1, model2, features1, features2

# def predict_cardiac_disease(input_values, model1, model2, features1, features2):
#     medical_input_len = len(features1)
#     lifestyle_input_len = len(features2)

#     if len(input_values) != medical_input_len + lifestyle_input_len:
#         raise ValueError(f"Expected {medical_input_len + lifestyle_input_len} inputs, but got {len(input_values)}.")

#     medical_input = pd.DataFrame([input_values[:medical_input_len]], columns=features1)
#     lifestyle_input = pd.DataFrame([input_values[medical_input_len:]], columns=features2)

#     prob1 = model1.predict_proba(medical_input)[0][1]
#     prob2 = model2.predict_proba(lifestyle_input)[0][1]

#     if prob1 >= 0.95 or (prob1 >= 0.8 and prob2 >= 0.6) or (prob1 >= 0.7 and prob2 >= 0.8) or prob2 >= 0.95:
#         final_prediction = 1
#     else:
#         final_prediction = 0

#     return {
#         "Final Prediction": "Heart Disease Detected" if final_prediction == 1 else "No Heart Disease",
#         "Medical Model Confidence": prob1,
#         "Lifestyle Model Confidence": prob2
#     }
import joblib
import pandas as pd
import os
import time

def load_models():
    try:
        model1_path = os.path.join(os.path.dirname(__file__), "model1_naive_bayes.pkl")
        model2_path = os.path.join(os.path.dirname(__file__), "model2_gb.pkl")
        features1_path = os.path.join(os.path.dirname(__file__), "model1_features.pkl")
        features2_path = os.path.join(os.path.dirname(__file__), "model2_features.pkl")
        
        if not all(os.path.exists(p) for p in [model1_path, model2_path, features1_path, features2_path]):
            missing = [p for p in [model1_path, model2_path, features1_path, features2_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Files not found: {missing}")
        
        start_time = time.time()
        model1 = joblib.load(model1_path)
        print(f"model1_naive_bayes.pkl loaded in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        model2 = joblib.load(model2_path)
        print(f"model2_gb.pkl loaded in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        features1 = joblib.load(features1_path)
        print(f"model1_features.pkl loaded in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        features2 = joblib.load(features2_path)
        print(f"model2_features.pkl loaded in {time.time() - start_time:.2f} seconds")
        
        return model1, model2, features1, features2
    except Exception as e:
        raise Exception(f"Failed to load models: {str(e)}")

def predict_cardiac_disease(input_values, model1, model2, features1, features2):
    medical_input_len = len(features1)
    lifestyle_input_len = len(features2)

    if len(input_values) != medical_input_len + lifestyle_input_len:
        raise ValueError(f"Expected {medical_input_len + lifestyle_input_len} inputs, but got {len(input_values)}.")

    medical_input = pd.DataFrame([input_values[:medical_input_len]], columns=features1)
    lifestyle_input = pd.DataFrame([input_values[medical_input_len:]], columns=features2)

    prob1 = model1.predict_proba(medical_input)[0][1]
    prob2 = model2.predict_proba(lifestyle_input)[0][1]

    if prob1 >= 0.95 or (prob1 >= 0.8 and prob2 >= 0.6) or (prob1 >= 0.7 and prob2 >= 0.8) or prob2 >= 0.95:
        final_prediction = 1
    else:
        final_prediction = 0

    return {
        "Final Prediction": "Heart Disease Detected" if final_prediction == 1 else "No Heart Disease",
        "Medical Model Confidence": prob1,
        "Lifestyle Model Confidence": prob2
    }
