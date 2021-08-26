import streamlit as st
import sys

# Update to web version after presentation

sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import streamlit_helper

st.set_page_config(layout='wide')
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('', ['Model', 'Plots', 'Summary', 'Worst Delinquency', 'Pipeline'])

if page == 'Model':
    streamlit_helper.model()

elif page == 'Plots':
    streamlit_helper.plots()
    
elif page == 'Summary':
    streamlit_helper.summary()

elif page == 'Worst Delinquency':
    streamlit_helper.worst_delinquency()
    
elif page == 'Pipeline':
    streamlit_helper.pipeline_example()

