import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)

def train_model():
    df = pd.read_csv("./milknew.csv")
    df_original = df.copy(deep=True)
    
    df_original['Grade'] = pd.factorize(df_original['Grade'])[0]

    X = df_original.copy()
    y = X.pop('Grade')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    print(X_train.shape, X_test.shape)
    
    forest = RandomForestClassifier(n_estimators=200, random_state=123)
    forest.fit(X_train.values, y_train.values)
    
    return forest
  
def result(pred):
    if pred == 0:
        return "High"
    elif pred == 1:
        return "Low"
    else:
        return "Medium"
    
def Temp(temp):
    if temp == "Cold":
        return 35
    elif temp == "Lukewarm":
        return 45
    else:
        return 70

def Colour(col):
    if col == "Yes":
        return 246
    else:
        return 254
    
def Odor(od):
    if od == "Yes":
        return 1
    else:
        return 0
    
def Turbidity(tb):
    if tb == "Yes":
        return 1
    else:
        return 0
    
def Taste(tst):
    if tst == "Yes":
        return 1
    else:
        return 0 
    
def pH(ph):
    if ph == "Yes":
        return 8.5
    else:
        return 6.6   
    
def Fat(ft):
    if ft == "Yes":
        return 0
    else:
        return 1     
       
st.sidebar.success("Select a Survey above.")
    
    
st.title('Predicting Milk Quality')

forest = train_model()

OptionPh = st.selectbox(
    'Does it have a soury taste?',
    ('No', 'Yes'))

OptionTemperature = st.selectbox(
    'How warm is your milk?',
    ('Cold', 'Lukewarm', 'Warm'))

OptionTaste = st.selectbox(
    'Does it taste different?',
    ('No', 'Yes'))

OptionOdor = st.selectbox(
    'Does the milk Smell?',
    ('No', 'Yes'))

Optionfat = st.selectbox(
    'is the milk chunky?',
    ('No', 'Yes'))

OptionColour = st.selectbox(
    'Does the milk look discoloured?',
    ('No', 'Yes'))


pred = forest.predict([[
    pH(OptionPh), 
    Temp(OptionTemperature), 
    Taste(OptionTaste), 
    Odor(OptionOdor), 
    Fat(Optionfat), 
    1, 
    Colour(OptionColour)]])

result = result(pred) 

st.text('The predicted quality is: ' + result) 
    