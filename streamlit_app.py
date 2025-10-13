# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# --- Load the Saved Model and Scaler ---
# We use a Streamlit cache decorator to only load the model once
@st.cache_resource
def load_model():
    with open('stock_predictor.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model()

# --- App Title and Description ---
st.title("Tesla Stock Price Movement Predictor 🚗⚡")
st.markdown("Enter today's stock data to forecast if the price will **rise** or **fall** tomorrow.")

# --- Create a Sidebar for User Input ---
st.sidebar.header("Input Today's Stock Data")

# Use number_input for user to enter values
open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=132.44, step=0.01)
close_price = st.sidebar.number_input("Close Price", min_value=0.0, value=132.42, step=0.01)
high_price = st.sidebar.number_input("High Price", min_value=0.0, value=134.76, step=0.01)
low_price = st.sidebar.number_input("Low Price", min_value=0.0, value=129.99, step=0.01)
month = st.sidebar.number_input("Current Month (1-12)", min_value=1, max_value=12, value=10, step=1)

# --- Prediction Logic ---
if st.sidebar.button("Predict Movement"):
    # Re-create the features exactly as in your notebook
    open_close_diff = open_price - close_price
    low_high_diff = low_price - high_price
    is_quarter_end = 1 if month % 3 == 0 else 0

    # Put the features into a numpy array
    features_to_predict = np.array([[open_close_diff, low_high_diff, is_quarter_end]])
    
    # Scale the features
    scaled_features = scaler.transform(features_to_predict)

    # Make the prediction
    prediction = model.predict(scaled_features)
    
    # --- Display the Result ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("The stock price is predicted to GO UP tomorrow! 📈")
    else:
        st.error("The stock price is predicted to GO DOWN tomorrow. 📉")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Streamlit & XGBoost")