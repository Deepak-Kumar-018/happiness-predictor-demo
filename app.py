# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
# Make sure 'happiness_model.joblib' is in the same directory as app.py
try:
    model = joblib.load('happiness_model.joblib')
except FileNotFoundError:
    st.error("Model file (happiness_model.joblib) not found. Make sure it's in the app directory.")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# App title
st.title("Student Happiness Predictor")
st.write("Enter the number of hours a student studied to predict their happiness score.")
st.write("(Based on a very simple demo model!)")

# Input field for hours studied
# Using st.number_input for better control over numeric input
hours_studied = st.number_input(
    "Hours Studied:",
    min_value=0.0,  # Minimum possible hours
    max_value=24.0, # Maximum possible hours (can adjust)
    value=5.0,      # Default value
    step=0.5        # Step for increment/decrement buttons
)

# Prediction button
if st.button("Predict Happiness"):
    if model:
        # Prepare the input for the model (needs to be a 2D array)
        input_features = np.array([[hours_studied]])
        
        # Make prediction
        prediction = model.predict(input_features)
        
        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"Predicted Happiness Score: **{prediction[0]:.2f}**")
        
        # Simple interpretation (optional)
        if prediction[0] > 80:
            st.write("This student is predicted to be very happy! 😊")
        elif prediction[0] > 60:
            st.write("This student is predicted to be moderately happy. 🙂")
        else:
            st.write("This student might need a break or some encouragement. 😐")
    else:
        st.write("Model is not loaded. Cannot make prediction.")

st.sidebar.info("This is a demo app using a Linear Regression model.")
st.sidebar.markdown("Created for demonstration purposes.")