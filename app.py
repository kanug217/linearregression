import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("data.csv")

# Prepare the features and target
x = df[['hoursstudied']]
y = df['examscore']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Streamlit app UI
st.title("📘 Exam Score Predictor")
st.write("Enter hours studied to predict the exam score.")

# Input from user
hours = st.number_input("Hours studied:", min_value=0.0, step=0.1)

# Predict on button click
if st.button("Predict Score"):
    predicted_score = model.predict([[hours]])[0]
    st.success(f"🎯 Predicted Score: {predicted_score:.2f}")

# Display the sample data
st.write("### Sample Training Data")
st.dataframe(df)