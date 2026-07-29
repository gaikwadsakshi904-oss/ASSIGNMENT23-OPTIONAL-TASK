import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Assignment 23 ML App",
    layout="wide"
)

st.title("Assignment 23 - Machine Learning Models")

page = st.sidebar.radio(
    "Module",
    [
        "Bike Price Prediction",
        "Employee Attrition Prediction"
    ]
)

# ------------------------------
# Bike Price Prediction
# ------------------------------

if page == "Bike Price Prediction":

    model = joblib.load("linear_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    encoders = joblib.load("bike_encoders.pkl")

    st.header("🚲 Used Bike Price Prediction")

    values = {}

    for col in columns:

        if col in encoders:

            option = st.selectbox(
                col,
                list(encoders[col].classes_)
            )

            values[col] = int(
                encoders[col].transform([option])[0]
            )

        else:

            values[col] = st.number_input(
                col,
                value=0.0
            )

    if st.button("Predict Bike Price"):

        input_data = pd.DataFrame([values])

        input_data = input_data[columns]

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Selling Price : ₹ {prediction[0]:,.2f}"
        )

# ------------------------------
# Employee Attrition Prediction
# ------------------------------

else:

    st.header("👨‍💼 Employee Attrition Prediction")

    algorithm = st.selectbox(
        "Select Algorithm",
        [
            "Logistic Regression",
            "KNN",
            "Naive Bayes"
        ]
    )

    if algorithm == "Logistic Regression":

        model = joblib.load("logistic_model.pkl")

    elif algorithm == "KNN":

        model = joblib.load("knn_model.pkl")

    else:

        model = joblib.load("naive_bayes_model.pkl")

    scaler = joblib.load("logistic_scaler.pkl")

    columns = joblib.load("hr_columns.pkl")

    encoders = joblib.load("hr_encoders.pkl")

    values = {}

    for col in columns:

        if col in encoders:

            option = st.selectbox(
                col,
                list(encoders[col].classes_),
                key=col
            )

            values[col] = int(
                encoders[col].transform([option])[0]
            )

        else:

            values[col] = st.number_input(
                col,
                value=0.0,
                key=col
            )

    if st.button("Predict Attrition"):

        input_data = pd.DataFrame([values])

        input_data = input_data[columns]

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.error("Employee is likely to leave the company.")

        else:

            st.success("Employee is likely to stay in the company.")