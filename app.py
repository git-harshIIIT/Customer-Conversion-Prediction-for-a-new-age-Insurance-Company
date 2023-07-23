import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('model.pkl')

# Function to map categorical features back to their original values
def map_back(column, mapping):
    if isinstance(mapping, dict):
        reverse_mapping = {v: k for k, v in mapping.items()}
        return column.map(reverse_mapping)
    else:
        return column

# Custom CSS styles for the app
st.markdown(
    """
    <style>
    .header {
        background-color: #33adff;
        color: white;
        padding: 0.5rem;
        text-align: center;
        font-size: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .button {
        background-color: #33adff;
        color: white;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
    }
    .prediction {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    st.title('Customer Insurance Subscription Prediction')
    st.markdown('<p class="header">Customer Details</p>', unsafe_allow_html=True)

    # Read the dataset
    dataset = pd.read_csv('train.csv')
    dataset = dataset.dropna(axis=0)

    # Map categorical features to numeric values
    mapping = {
        # ... (your mapping code here)
    }
    dataset.replace(mapping, inplace=True)

    # Encode 'job' feature using LabelEncoder
    le = LabelEncoder()
    dataset['job'] = le.fit_transform(dataset['job'])

    # Inverse transform function to map numeric values back to job names
    def inverse_transform_job(code):
        return le.inverse_transform([code])[0]

    # Get user inputs for features
    job = st.selectbox('Job', dataset['job'].unique(), format_func=inverse_transform_job)
    # ... (rest of your code)

    # Add a "Predict" button
    if st.button('Predict', key='predict_button'):
        # Prepare the input data for prediction
        data = pd.DataFrame({
            'job': [job],
            'marital': [marital],
            'education_qual': [education_qual],
            'call_type': [call_type],
            'prev_outcome': [prev_outcome],
            'mon': [mon],
            'age': [age],
            'day': [day],
            'dur': [dur],
            'num_calls': [num_calls]
        })

        # Make prediction using the loaded model
        prediction = model.predict(data)[0]

        # Display the prediction
        st.markdown('<p class="prediction">Prediction:</p>', unsafe_allow_html=True)
        if prediction == 0:
            st.write('No, The customer is highly unlikely to subscribe to the insurance.')
        else:
            st.write('Yes, The customer is highly likely to subscribe to the insurance.')

# Run the app
if __name__ == '__main__':
    main()
