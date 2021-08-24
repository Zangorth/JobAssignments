import streamlit as st
import pandas as pd
import sys
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import helper

st.set_page_config(layout='wide')
st.title('DriveTime')
st.header('Pipeline Example')

file = st.sidebar.file_uploader('Customer Data File for Prediction', type=['.csv'], accept_multiple_files=False)

if file != None:
    panda = pd.read_csv(file, encoding='utf8')
    
    st.write('')
    st.write('Input')
    st.dataframe(panda)

    probs = helper.pipe('account-defaults.csv')
    ranks = helper.pipe('account-defaults.csv', output='rank')
    panda = pd.DataFrame({'probability': probs, 'rank': ranks})
    
    st.write('')
    st.write('Output')    
    st.dataframe(panda)
    
    download_path = st.sidebar.text_input('Download Path:', f'C:/Users/{os.getlogin()}/Downloads')
    download = st.sidebar.button('DOWNLOAD')
    
    if download:
        panda.to_csv(f'{download_path}/{file.name.replace(".csv", "")} probabilities and ranking.csv', encoding='utf-8', index=False)