# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:17:54 2023

@author: fruit
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

#loading saved model
loaded_model=pickle.load(open('C:/Users/fruit/Downloads/trained_model.sav' , 'rb'))

#creating prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
      print('The person is not diabetic')
    else:
      print('The person is diabetic')
def main():
    #giving title
    st.title('Diabetes Prediction')
   
    #giving input data from user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Value')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    #code for prediction
    diagnosis=""
    if st.button("Diabetes Test Results"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI, DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__=='__main__':
    main()

    