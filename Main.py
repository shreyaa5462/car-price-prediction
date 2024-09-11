import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split #for splitting the data
from sklearn.linear_model import LinearRegression #for the model
from sklearn.linear_model import Lasso
from sklearn import metrics


def main():
    # Set background color using HTML
    st.set_page_config(page_title='Car Price Prediction', page_icon=':car:', layout='centered')
    page_bg_image = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1560807707-8cc77767d783");
    }
    </style>
    """

    st.markdown(page_bg_image, unsafe_allow_html=True)

    st.write(f"<h1 style='color : red; text-align:center; padding-bottom :80px;'>Car Price Prediction</h3>",unsafe_allow_html=True)
    
    #load the data
    car_Dataset=pd.read_csv('car data.csv')
    #Encoding the categorical data

    #Encoding the "Fuel_Type" Column
    car_Dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

    #Encoding the "Sellet_Type" Column
    car_Dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

    #Encoding the "Transmission" Column
    car_Dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

    X=car_Dataset.drop(['Car_Name','Selling_Price'],axis=1) #mention axis because we are dropping columns
    Y=car_Dataset['Selling_Price']


    # Splitting the Data into Training Data and Test data
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)

    #Loading the Linear Regression Model
    lin_Reg_Model=LinearRegression() #Creating the Model

    #Training the Model
    lin_Reg_Model.fit(X_train,Y_train)

    #Predicting the Training Data
    train_data_prediciton=lin_Reg_Model.predict(X_train)

    #Predicting the Test Data

    st.subheader("Please Enter  the  Details  to Predict the Price of the Car")

    # Feature - 1
    year=st.selectbox(label="Year of Buying the Car",options=list(range(2000,2025)))

    # Feature - 2
    cost=st.number_input("The Cost of the Car in Lakhs",min_value=None,step=None,format=None)

    # Feature - 3
    km_Driven=st.number_input("Total No of Kilometers Driven :")

    # Feature - 4
    fuel_Type=st.selectbox("Fuel Type",('Petrol','Diesel','CNG'))
    if fuel_Type=='Petrol':
        fuel_Type=0
    elif fuel_Type=='Diesel':
        fuel_Type=1
    else:
        fuel_Type=2

    # Feature - 5
    seller_Type=st.selectbox("Seller Type",('Dealer','Individual'))
    if seller_Type=='Dealer':
        seller_Type=0
    else:
        seller_Type=1

    # Feature - 6
    transmission=st.selectbox("Transmission",('Manual','Automatic'))
    if transmission=='Manual':
        transmission=0
    else:
        transmission=1

    # Feature - 7
    owner=st.selectbox(("Owner"),('First Owner','Second Owner'))
    if owner=='First Owner':
        owner=0
    elif owner=='Second Owner':
        owner=1

    X=np.array([[year,cost,km_Driven,fuel_Type,seller_Type,transmission,owner]])


    test_data_prediciton=lin_Reg_Model.predict(X)

    if st.button("Predict",type='primary'):
        with st.spinner('Predicting ...'):
            time.sleep(3)

    
        if test_data_prediciton[0]<0:
            st.write(f"<h3 style='color :red;'>Please Enter the Correct Details</h3>",unsafe_allow_html=True)
        else:
            price=test_data_prediciton[0]*100000
            price=round(price,2)
            st.write(f"<h3 style='color :red;'>The Predicted Price of the Car is :{price} Rupees</h3>",unsafe_allow_html=True)
            num=random.randint(0,1)
            if num==0:
                st.balloons()
            if num==1:
                st.snow()
            st.info("NOTE : These are Just Predicted Values based on the Previous Data and Not Exact Values ",icon='⚠️')
            time.sleep(3)
            st.toast("Copy Rights ©️ shreyaa5462",icon='🟥')
            time.sleep(4)

if __name__ == '__main__':
    main()
