import streamlit as st
from joblib import load

st.title("Let's begin!!")

clf = load("DT.joblib")

LABELS = ['setosa', 'versicolor', 'virginica']

sp_l = st.slider('sepal length (cm)', min_value=0, max_value=10)
sp_w = st.slider('sepal width (cm)', min_value=0, max_value=10)
pe_l = st.slider('petal length (cm)', min_value=0, max_value=10)
pe_w = st.slider('petal width (cm)', min_value=0, max_value=10)

predict = clf.predict([[sp_l, sp_w, pe_l, pe_w]])

st.write(LABELS[predict[0]])