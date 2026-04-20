writefile streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(layout="wide", page_title="Real Estate Investment Advisor")

# --- Constants ---
HIGH_GROWTH_CITIES = ['mumbai', 'bangalore', 'bengaluru', 'pune', 'hyderabad']

# --- Helper Functions ---
@st.cache_resource
def load_model_and_artifacts():
    models_path = 'models'
    reg_model = pickle.load(open(os.path.join(models_path, 'reg_model.pkl'), 'rb'))
    clf_model = pickle.load(open(os.path.join(models_path, 'clf_model.pkl'), 'rb'))
    features = pickle.load(open(os.path.join(models_path, 'features.pkl'), 'rb'))
    encoders = pickle.load(open(os.path.join(models_path, 'encoders.pkl'), 'rb'))
    return reg_model, clf_model, features, encoders

reg_model, clf_model, features, encoders = load_model_and_artifacts()

def preprocess_input(input_df, features, encoders):
    processed_data = pd.DataFrame(index=[0])

    for feature in features:
        if feature in input_df.columns:
            processed_data[feature] = input_df[feature].values[0]
        else:
            # Handle features not directly from user input (e.g., encoded categoricals)
            if feature.endswith('_encoded'):
                original_col = feature.replace('_encoded', '')
                if original_col in input_df.columns and original_col in encoders:
                    # Ensure input value is in lower case for consistent encoding
                    val = input_df[original_col].values[0].lower()
                    encoder = encoders[original_col]
                    try:
                        processed_data[feature] = encoder.transform([val])[0]
                    except ValueError:
                        # Handle unseen labels by assigning a default or mode, for now using 0
                        processed_data[feature] = 0 # Or any other appropriate default
                else:
                    processed_data[feature] = 0 # Default for missing encoded features
            else:
                processed_data[feature] = 0 # Default for other missing features

    return processed_data

# --- Streamlit App --- 
st.title("🇮🇳 Real Estate Investment Advisor")
st.markdown("### Predict Property Profitability & Future Value")

st.subheader("Property Details Input")

# Organize input into columns
col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State", options=[enc.lower() for enc in encoders['State'].classes_])
    city = st.selectbox("City", options=[enc.lower() for enc in encoders['City'].classes_])
    locality = st.selectbox("Locality", options=[enc.lower() for enc in encoders['Locality'].classes_])
    property_type = st.selectbox("Property Type", options=[enc.lower() for enc in encoders['Property_Type'].classes_])
    bhk = st.slider("BHK", 1, 10, 2)

with col2:
    size_sqft = st.number_input("Size in SqFt", min_value=100, max_value=50000, value=1200)
    year_built = st.slider("Year Built", 1900, 2024, 2000)
    floor_no = st.slider("Floor Number", 0, 50, 1)
    total_floors = st.slider("Total Floors", 1, 100, 5)
    furnished_status = st.selectbox("Furnished Status", options=[enc.lower() for enc in encoders['Furnished_Status'].classes_])

with col3:
    nearby_schools = st.slider("Nearby Schools (km)", 0, 20, 2)
    nearby_hospitals = st.slider("Nearby Hospitals (km)", 0, 20, 2)
    public_transport = st.selectbox("Public Transport Accessibility", options=[enc.lower() for enc in encoders['Public_Transport_Accessibility'].classes_])
    parking_space = st.selectbox("Parking Space", options=[enc.lower() for enc in encoders['Parking_Space'].classes_])
    security = st.selectbox("Security", options=[enc.lower() for enc in encoders['Security'].classes_])
    facing = st.selectbox("Facing", options=[enc.lower() for enc in encoders['Facing'].classes_])
    owner_type = st.selectbox("Owner Type", options=[enc.lower() for enc in encoders['Owner_Type'].classes_])
    availability_status = st.selectbox("Availability Status", options=[enc.lower() for enc in encoders['Availability_Status'].classes_])
    # price_in_lakhs = st.number_input("Current Price in Lakhs (for benchmark)", min_value=1.0, max_value=10000.0, value=50.0)

# Create a dictionary from inputs
input_data = {
    'State': state,
    'City': city,
    'Locality': locality,
    'Property_Type': property_type,
    'BHK': bhk,
    'Size_in_SqFt': size_sqft,
    'Year_Built': year_built,
    'Floor_No': floor_no,
    'Total_Floors': total_floors,
    'Furnished_Status': furnished_status,
    'Nearby_Schools': nearby_schools,
    'Nearby_Hospitals': nearby_hospitals,
    'Public_Transport_Accessibility': public_transport,
    'Parking_Space': parking_space,
    'Security': security,
    'Facing': facing,
    'Owner_Type': owner_type,
    'Availability_Status': availability_status,
    # 'Price_in_Lakhs': price_in_lakhs, # Not used as input for prediction directly, calculated if needed
}

# Add derived features needed for encoding but not directly in `X`
input_data['Age_of_Property'] = 2025 - year_built # Assuming current year for age calculation

# Create a DataFrame from input data
input_df = pd.DataFrame([input_data])

# Preprocess input for models
processed_input = preprocess_input(input_df, features, encoders)

# Make predictions
if st.button("Predict"):    
    # Classification Prediction
    clf_prediction = clf_model.predict(processed_input)
    is_good_investment = "Yes" if clf_prediction[0] == 1 else "No"

    # Regression Prediction
    # Calculate Price_in_Lakhs for future price calculation if needed, 
    # but reg_model uses its own features derived from input.
    # The original notebook derived Future_Price_5Y_Lakhs from Price_in_Lakhs directly. 
    # Here we directly predict it using the features X.
    future_price_pred = reg_model.predict(processed_input)[0]

    st.subheader("Prediction Results")
    
    colA, colB = st.columns(2)

    with colA:
        st.success(f"**Good Investment (Classification):** {is_good_investment}")
        if is_good_investment == "Yes":
            st.balloons()
        else:
            st.warning("Consider other options or re-evaluate criteria.")

    with colB:
        st.info(f"**Estimated Future Price (5 Years):** ₹{future_price_pred:,.2f} Lakhs")
        st.markdown("*(Note: This is a direct prediction based on the trained regression model.)*")


st.markdown("""
--- 
### How to Run this Streamlit App:
1. Save the code above as `streamlit_app.py` in your working directory.
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the command: `streamlit run streamlit_app.py`
5. The app will open in your web browser.
""")
