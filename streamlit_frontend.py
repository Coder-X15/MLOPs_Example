import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
# Set the page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🚀"
    )

st.title("Iris Flower Species Classifier")
st.write("""
This application classifies iris flower species based on user-provided measurements.
        """)
st.write("Please enter the following measurements:")
# Input fields for iris flower measurements
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Classify"):
    # send request to the backend API
    import requests
    payload = {
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_wid]t": [petal_width]
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        result = result['predictions'][0]
        match result:
            case 0:
                result = "setosa"
            case 1:
                result = "versicolor"
            case 2:
                result = "virginica"

        st.success(f"The predicted species is: iris {result}")
    else:
        st.error("Error in classification. Please try again.")

st.write("Developed by Sam Ruben Abraham")