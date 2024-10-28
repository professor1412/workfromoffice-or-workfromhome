import streamlit as st
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))


st.title("Predict Target Value")


age = st.number_input("Age", min_value=0)
occupation = st.selectbox("Occupation", options=["Tutor", "HR", "Engineer", "Recruiter", "Business","Marketing","Manager","Other"])
gender = st.selectbox("Gender", options=["Male", "Female"])
same_office_home_location = st.selectbox("Same Office Home Location ", options=["Yes","No"])
kids = st.selectbox("Having Kids ?", options=["Yes","No"])
rm_save_money = st.selectbox("RM Save Money",options= ["Yes","No"])
rm_quality_time = st.selectbox("RM Quality Time", options= ["Yes","No"])
rm_better_sleep = st.selectbox("RM Better Sleep", options= ["Yes","No"])
calmer_stressed = st.selectbox("Calmer Stressed", options= ["CALMER","STRESSED"])
rm_professional_growth = st.number_input("RM Professional Growth (1-5)", min_value=1, max_value=5)
rm_lazy = st.number_input("RM Lazy (1-5)", min_value=1, max_value=5)
rm_productive = st.number_input("RM Productive", min_value=1, max_value=5)
digital_connect_sufficient = st.selectbox("Digital Connect Sufficient", options=["Yes","No"])
rm_better_work_life_balance = st.number_input("RM Better Work-Life Balance (1-5)", min_value=1, max_value=5)
rm_improved_skillset = st.number_input("RM Improved Skillset", min_value=1, max_value=5)
rm_job_opportunities = st.selectbox("RM Job Opportunities", options=["Yes","No","Not sure"])


input_data = np.array([age,
                       occupation,
                       gender,
                       same_office_home_location,
                       kids,
                       rm_save_money,
                       rm_quality_time,
                       rm_better_sleep,
                       calmer_stressed,
                       rm_professional_growth,
                       rm_lazy,
                       rm_productive,
                       digital_connect_sufficient,
                       rm_better_work_life_balance,
                       rm_improved_skillset,
                       rm_job_opportunities]).reshape(1, -1)


input_data[0][1] = {
    "Tutor" :0,
    "HR": 1,
    "Engineer": 2,
    "Recruiter": 3,
    "Business": 4,
    "Marketing": 5,
    "Manager": 6,
    "Other": 7
}.get(occupation, 7)
input_data[0][2] = {'Male': 1, 'Female': 0}.get(gender)
input_data[0][3] = {'Yes': 1, 'No': 0}.get(same_office_home_location)
input_data[0][4] = {'Yes': 1, 'No': 0}.get(kids)
input_data[0][5] = {'Yes': 1, 'No': 0}.get(rm_save_money)
input_data[0][6] = {'Yes': 1, 'No': 0}.get(rm_quality_time)
input_data[0][7] = {'Yes': 1, 'No': 0}.get(rm_better_sleep)
input_data[0][8] = {'CALMER': 1, 'STRESSED': 0}.get(calmer_stressed)
input_data[0][12] = {'Yes': 1, 'No': 0}.get(digital_connect_sufficient)
input_data[0][15] = {'Yes': 1, 'No': 0,'Not sure': 2}.get(rm_job_opportunities)



# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Predict the target value
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0]==1:
        st.write("Employee wants to work from Home")
    elif prediction[0]==0:
        st.write("Employee wants to work from office")

