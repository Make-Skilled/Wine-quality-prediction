import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('logistic_regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title("🍷 Wine Quality Predictor")

st.write("Enter the chemical properties of wine to predict its quality.")

# User Input Fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001, format="%.6f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, step=1)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, step=1)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.6f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Collect input features into an array
input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                            pH, sulphates, alcohol]])

# Predict using the model
if st.button("Predict Quality"):
    prediction = model.predict(input_features)[0]

    # Categorize Quality
    if prediction <= 5:
        quality_label = "❌ Bad Quality"
        advice = (
            "🔹 Reduce volatile acidity.\n"
            "🔹 Check sulfur dioxide levels to prevent oxidation.\n"
            "🔹 Balance acidity and residual sugar.\n"
            "🔹 Improve fermentation process."
        )
        st.error(f"Predicted Wine Quality: {prediction} ({quality_label})")
        st.warning("⚠️ Precautions to Improve Wine Quality:")
        st.write(advice)

    elif prediction == 6:
        quality_label = "✅ OK Quality"
        st.info(f"Predicted Wine Quality: {prediction} ({quality_label})")
        st.write("🔹 Decent wine, but slight improvements can enhance its quality!")

    else:
        quality_label = "🏆 Best Quality"
        st.success(f"Predicted Wine Quality: {prediction} ({quality_label})")
        st.balloons()
        st.write("🎉 Congratulations! Your wine meets premium quality standards!")

