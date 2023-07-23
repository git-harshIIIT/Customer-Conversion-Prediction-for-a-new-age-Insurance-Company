
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

# Streamlit app
def main():
    st.title('Insurance Subscription Prediction')

    # Read the dataset
    dataset = pd.read_csv('train.csv')
    dataset = dataset.dropna(axis=0)

    # Map categorical features to numeric values
    mapping = {
        'y': {'no': 0, 'yes': 1},
        'mon': {'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'sep': 9},
        'education_qual': {'tertiary': 3, 'secondary': 2, 'unknown': 0, 'primary': 1},
        'marital': {'married': 24, 'single': 16, 'divorced': 32},
        'call_type': {'unknown': 0, 'cellular': 1, 'telephone': 2},
        'prev_outcome': {'unknown': 2, 'failure': 0, 'other': 3, 'success': 1}
    }
    dataset.replace(mapping, inplace=True)

    # Encode 'job' feature using LabelEncoder
    le = LabelEncoder()
    dataset['job'] = le.fit_transform(dataset['job'])

    # Get user inputs for features
    st.subheader('Enter the Customer Details:')
    job = st.selectbox('Job', dataset['job'].unique(), format_func=lambda x: map_back(pd.Series([x]), le.classes_)[0])
    marital = st.selectbox('Marital Status', dataset['marital'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['marital'])[0])
    education_qual = st.selectbox('Education Qualification', dataset['education_qual'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['education_qual'])[0])
    call_type = st.selectbox('Call Type', dataset['call_type'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['call_type'])[0])
    prev_outcome = st.selectbox('Previous Outcome', dataset['prev_outcome'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['prev_outcome'])[0])
    mon = st.selectbox('Month', dataset['mon'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['mon'])[0])
    age = st.slider('Age', min_value=0, max_value=100, value=30)
    day = st.slider('Day', min_value=1, max_value=31, value=1)
    dur = st.slider('Duration', min_value=0, max_value=1000, value=200)
    num_calls = st.slider('Number of Calls', min_value=0, max_value=20, value=5)

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
    st.subheader('Prediction:')
    if prediction == 0:
        st.write('The customer will not subscribe to the insurance.')
    else:
        st.write('The customer will subscribe to the insurance.')

# Run the app
if __name__ == '__main__':
    main()
