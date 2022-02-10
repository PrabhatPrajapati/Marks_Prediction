import streamlit as st
import joblib
import matplotlib.pyplot as plt
st.set_page_config(page_title="student marks prediction",page_icon="😎")
st.title("Student Marks Prediction")
st.write("""
Made by *Prabhat Kumar*
""")
prabhat=joblib.load("student_marks_prediction_model.pkl")
exp=st.sidebar.slider("Hours",1,10,2)
st.write(f"Hours",exp)
y_pred=prabhat.predict([[exp]]).round(3)
st.write(f"Obtained Marks is: ",float(y_pred))




