import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.features.build_features import create_dummy_vars, separate_features_target
from src.data.make_dataset import load_data
from src.models.train_model import train_decision_tree
from src.models.predict_model import predict_decision_tree

# Load dataset and preprocess
df = load_data()
df = create_dummy_vars(df)
X, y = separate_features_target(df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to handle large discrepancies in feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Decision Tree model (this is usually done once and pickled)
dt_model = train_decision_tree(X_train_scaled, y_train)

# Save the trained Decision Tree model using pickle (only needs to be done once)
with open('src/models/dt_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

# Load the trained model (this should be done only once at the start of the app)
with open('src/models/dt_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

# Streamlit User Interface
st.title("Real Estate Price Prediction")

# Get user input for property features
property_type = st.selectbox("Property Type", ["Bunglow", "Condo"])
sqft = st.number_input("Square Footage", min_value=300, max_value=10000, value=1500)  # Reasonable boundaries
beds = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)  # Reasonable boundaries
baths = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)  # Reasonable boundaries
lot_size = st.number_input("Lot Size", min_value=100, max_value=50000, value=5000)  # Reasonable boundaries

# Feature input dictionary
user_input = {
    'property_type_Bunglow': 1 if property_type == "Bunglow" else 0,
    'property_type_Condo': 1 if property_type == "Condo" else 0,
    'sqft': sqft,
    'beds': beds,
    'baths': baths,
    'lot_size': lot_size
}

# Function to align input features with training data columns
def align_columns(X_train, input_df):
    """
    Align the input data columns with the training data columns.
    If any columns are missing in the input, add them with default values (0).
    """
    train_columns = X_train.columns  # Get the list of columns from the training data

    # Ensure that input_df has the same columns as the training data
    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing column with default value (0)

    # Reorder columns to match the training data
    input_df = input_df[train_columns]
    return input_df

# Convert user input dictionary to DataFrame
input_df = pd.DataFrame(user_input, index=[0])

# Align the columns of the user input to match the training data
input_df = align_columns(X_train, input_df)

# Scale the input data using the same scaler as the training data
input_scaled = scaler.transform(input_df)

# Debugging: Print user input and check if the scaling is correct
st.write("User Input Features:")
st.write(input_df)

# Add a submit button to trigger prediction
if st.button("Submit"):
    # Make prediction based on the updated user input
    prediction = predict_decision_tree(dt_model, input_scaled)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")
