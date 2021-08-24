import streamlit as st
import sys

sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import streamlit_helper

st.sidebar.header('Navigation')
page = st.sidebar.selectbox('', ['Model', 'Plots', 'Worst Delinquency', 'Pipeline'])

if page == 'Model':
    streamlit_helper.model()

elif page == 'Plots':
    streamlit_helper.plots()

elif page == 'Worst Delinquency':
    streamlit_helper.worst_delinquency()
    
elif page == 'Pipeline':
    streamlit_helper.pipeline_example()

