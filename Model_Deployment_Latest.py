#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

loaded_model = pickle.load(open(r'trained_model.sav','rb'))

def check(input_data):

    array_input = np.array(input_data)

    reshaped_input = array_input.reshape(1,-1)

    prediction = loaded_model.predict(reshaped_input)

    return "{:.2f}".format(prediction[0])


def main():
    st.title("House Price Prediciton")
    # Everything in Average of that particular Area except population
    Income = st.number_input("Average Annual Income of your Area in USD")
    
    Neigh_House_Age = st.number_input("Average House Age of Neighbours of your Area in Years")
    
    No_of_Rooms = st.number_input("Average Numbers of Rooms of your Area")
    
    No_of_BedRooms = st.number_input("Average Numbers of BedRooms of your Area")
    
    Population = st.number_input("Population of your Area")
    
    pred = ""
    if st.button("Click Here for Price Prediction of House"):
        pred = check([Income, Neigh_House_Age, No_of_Rooms, No_of_BedRooms, Population])
        
    st.success(f"The Predicted Price is {pred} $")
    
if __name__=='__main__':
    main()

