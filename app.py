import numpy as np
import streamlit as st
import pickle

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("The Laptop Predictor WebApp")
st.subheader("Get your laptop price as per your requirements")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# ram
ram = st.selectbox('RAM size(GB)', df['Ram'].unique())

# weight
weight = st.number_input("Weight")

# touchscreen
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

# IPS
IPS = st.selectbox('IPS panel', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',[
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2800x1800', '2560x1600', '2560x1400', '2304x1440'
])

# cpu
cpu = st.selectbox('Processors',df['Cpu brand'].unique())

# hdd and ssd

HDD = st.selectbox('HDD (GB)', [0, 512, 1024, 2048])
SSD = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024, 2048])

# GPU
Gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

# os
OS = st.selectbox('OS',df['OS'].unique())

if st.button("Predict the Price"):

    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if IPS == "Yes":
        IPS = 1
    else:
        IPS = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))**0.5 / (screen_size)

    query = np.array([company, type, ram, weight, touchscreen, IPS, ppi,cpu, HDD, SSD, Gpu, OS])

    query = query.reshape(1, 12) # there are 1 rows and 12 columns

    # st.title(pipe.predict(query))
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
