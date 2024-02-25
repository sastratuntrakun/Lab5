import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron

model = pickle.load(open('per_model-66130701725.sav','rb'))
st.title('Iris Species Prediction using Perceptron')

x1 = st.slider('Select Input1',0.0,10.0,6.7)
x2 = st.slider('Select Input2',0.0,10.0,3.0)
x3 = st.slider('Select Input3',0.0,10.0,5.2)
x4 = st.slider('Select Input4',0.0,10.0,2.3)

xnew = np.array([x1,x2,x3,x4]).reshape(1,-1)

predict = model.predict(xnew)
st.write("## Prediction Result")
st.write('Species: ',predict[0])