from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from db import Prediction
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns



st.title("Churn Prediction")

st.image("ccccccc.jpg")

st.success('''Churn prediction consists of detecting which customers are likely to cancel a subscription to a
            service based on how they use the service. ... Upload that data to a prediction service that automatically 
            creates a “predictive model.” Use the model on each current customer to predict whether they are at risk of
             leaving.''')




@st.cache()
def load_data(path):
    df= pd.read_csv(path)
    return df

df = load_data("DATASET/Churn_Modelling.csv")



def load_model(path = 'churn_prediction.h5'):
    model = tf.keras.models.load_model(path)
    with open('encoder.pk','rb') as file:
        genderenc = pickle.load(file)
    st.sidebar.info("Model Loaded Sucessfully.")
    st.sidebar.info("Upload Data for Prediction")
    return model,genderenc

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_results(name,features,result,output):
    try:
        db = opendb()
        p = Prediction(name=name,features=str(features),result=result,output=output)
        db.add(p)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False


if st.checkbox("About"):
    st.markdown("""
    Churn prediction consists of detecting which customers are likely to cancel a subscription to a service based
    on how they use the service. We want to predict the answer to the following question, asked for each current customer: 
    “Is this customer going to leave us within the next X months?” There are only two possible answers, yes or no, and it is
    what we call a binary classification task. Here, the input of the task is a customer and the output is the answer to the 
    question (yes or no).

    Being able to predict churn based on customer data has proven extremely valuable to big telecom companies.
    These are the process to predict churn:

    Gather historical customer data that you save to a CSV file.
    Upload that data to a prediction service that automatically creates a “predictive model.”
    Use the model on each current customer to predict whether they are at risk of leaving""" )



if  st.checkbox("Make Prediction"):
    model,genderEnc = load_model()

    name = st.text_input("Enter Customer Name")

    creditscore= st.number_input("enter cutomer credit score",min_value=0.0,max_value=1000.0,value=500.0)


    col1, col2 = st.beta_columns(2)

    with col1:
        gender = st.radio("Gender",("Male","Female"))
        

    with col2:
        age = st.number_input("Age", min_value=10,max_value=100,value=18)


    col3, col4 = st.beta_columns(2)
    
    with col3:
        tenure = st.number_input("Tenure",min_value=0,max_value=10,value=5)

    with col4:
        balance = st.number_input("Balance",min_value=0.0,max_value=250898.0,value=0.0)

    col5, col6 = st.beta_columns(2)
    
    with col5:
        numofproduct = st.number_input("NO. of Product",min_value=1.0,max_value=4.0,value=2.0)

    with col6:
        hascard = st.number_input("Has Cr Card",min_value=0.0,max_value=1.0,value=0.0)

    col7, col8 = st.beta_columns(2)
    
    with col5:
        IsActiveMember = st.number_input("Is Active Member",min_value=0.0,max_value=1.0,value=0.0)

        

    with col6:
        salary = st.number_input("Salary",min_value=11.58,max_value=199992.48,value=12.0)
        

    

    if st.sidebar.button('Predict')and name:
        gen = genderEnc.transform(np.array([[gender]]))[0]
        features = np.array([[creditscore,gen,age,tenure,balance,numofproduct,hascard,IsActiveMember,salary]])
        
        output = model.predict(features)[0][0]

        if output > 0.5:
            result = "Customer is likely to leave the bank."
            
        else:
            result = "This candidate will be a loyal customer."
        s = save_results(name,features, result, output)


        st.sidebar.success(result)
        if s:
        
            st.sidebar.success("Saved to Database")
        

    else:
        st.sidebar.warning("Fill all the details")

if st.checkbox("View Records"):
    db = opendb()
    results = db.query(Prediction).all()
    db.close()
    record = st.selectbox("select a cutomer Record",results)
    if record:
        st.write(record.name)
        st.write(record.result)
        st.write(f"{record.output:.2f}")
        st.write(record.created_on)


if st.checkbox("Analysis"):
    col = st.selectbox("Select a col for distribution graph",df.columns.tolist()+["Graph of training Dataset"])
    if col:
        if col == "Graph of training Dataset":
            st.image("train_graph.png")


        elif df[col].dtype == "int64" or df[col].dtype =="float" or df[col].dtype =="int" or df[col].dtype =="int32":
            fig,ax = plt.subplots(figsize=(12,6))
            df[col].plot(kind="hist",title=f'{col} Distribution',ax=ax)
            st.pyplot(fig)
    elif df [col].dtype =="object":
        fig,ax =plt.subplots(figsize =(12,6))
        sns.countplot(x=col, data=df,ax=ax)
        st.pyplot(fig)

