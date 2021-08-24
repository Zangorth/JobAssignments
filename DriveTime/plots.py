from matplotlib import pyplot as plt
import streamlit as st
import seaborn as sea
import pandas as pd
import numpy as np
import pickle
import sys
import re
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import helper

panda = pd.read_csv('account-defaults.csv', encoding='utf-8')

features = [col for col in panda.columns if col not in ['ID', 'FirstYearDelinquency']]
features.sort()

#st.set_page_config(layout='wide')
st.title('DriveTime')
st.header('Correlates of First Year Delinquency')

st.sidebar.subheader('Features')
feature_select = st.sidebar.selectbox('', features)

name = re.sub(r"(\w)([A-Z])", r"\1 \2", feature_select)

##############
# Univariate #
##############

st.subheader(f'{name} - Distribution')

to_plot = panda[feature_select].fillna(panda[feature_select].min()-panda[feature_select].std()*5)

sea.set_style('whitegrid')
uni_fig, uni_ax = plt.subplots(1, figsize=(7, 4))
sea.histplot(x=to_plot, stat='percent')
plt.axvline(panda[feature_select].mean(), ls='--', color='red')

st.pyplot(uni_fig)
plt.close()
st.write('')


#############
# Bivariate #
#############
panda['quantile'] = np.nan
for i in np.arange(0.1, 1.1, 0.1):
    mn, mx = panda[feature_select].quantile(i-0.1), panda[feature_select].quantile(i)
    
    if mn != mx:
        panda.loc[(panda[feature_select] >= mn) & (panda[feature_select] < mx), 'quantile'] = round(panda.loc[(panda[feature_select] >= mn) & (panda[feature_select] < mx), feature_select].mean(), 2)
    
    else:
        panda.loc[panda[feature_select] == mn, 'quantile'] = round(panda.loc[panda[feature_select] == mn, 'quantile'].mean(), 2)

name = re.sub(r"(\w)([A-Z])", r"\1 \2", feature_select)

st.subheader('Bivariate Relationship')

sea.set_style('whitegrid')
bi_fig, bi_ax = plt.subplots(1, figsize=(7, 4))
sea.pointplot(x='quantile', y='FirstYearDelinquency', hue='quantile', data=panda)
plt.ylabel('First Year Delinquency Rate')
plt.xlabel(name)
plt.legend().remove()

st.pyplot(bi_fig)
plt.close()
st.write('')

###############
# Predictions #
###############
reset = st.sidebar.button('SET TO AVERAGE')

if reset:
    st.session_state['na'] = int(round(panda['AgeNewestAutoAccount'].mean(), 0))
    st.session_state['oa'] = int(round(panda['AgeOldestAccount'].mean(), 0))
    st.session_state['oi'] = int(round(panda['AgeOldestIdentityRecord'].mean(), 0))
    st.session_state['aa'] = int(round(panda['AvgAgeAutoAccounts'].mean(), 0))
    st.session_state['it'] = int(round(panda['HasInquiryTelecomm'].mean(), 0))
    st.session_state['nd'] = int(round(panda['TotalAutoAccountsNeverDelinquent'].mean(), 0))
    st.session_state['ti'] = int(round(panda['TotalInquiries'].mean(), 0))
    st.session_state['wd'] = int(round(panda['WorstDelinquency'].mean(), 0))

na = st.sidebar.slider('Age Newest Auto Account', int(panda['AgeNewestAutoAccount'].min()), key='na')
oa = st.sidebar.slider('Age Oldest Auto Account', int(panda['AgeOldestAccount'].min()), int(panda['AgeOldestAccount'].max()), key='oa')
oi = st.sidebar.slider('Age Oldest Identity Record', int(panda['AgeOldestIdentityRecord'].min()), int(panda['AgeOldestIdentityRecord'].max()), key='oi')
aa = st.sidebar.slider('Average Age Auto Account', int(panda['AvgAgeAutoAccounts'].min()), int(panda['AvgAgeAutoAccounts'].max()), key='aa')
it = st.sidebar.slider('Has Inquiry Telecom', int(panda['HasInquiryTelecomm'].min()), int(panda['HasInquiryTelecomm'].max()), key='it')
nd = st.sidebar.slider('Total Auto Accounts Never Delinquenct', int(panda['TotalAutoAccountsNeverDelinquent'].min()), int(panda['TotalAutoAccountsNeverDelinquent'].max()), key='nd')
ti = st.sidebar.slider('Total Inquiries', int(panda['TotalInquiries'].min()), int(panda['TotalInquiries'].max()), key='ti')
wd = st.sidebar.slider('Worst Delinquency', int(panda['WorstDelinquency'].min()), int(panda['WorstDelinquency'].max()), key='wd')

typical_person = pd.DataFrame({'AgeOldestIdentityRecord': [oi],
                                'AgeOldestAccount': [oa],
                                'AgeNewestAutoAccount': [na],
                                'TotalInquiries': [ti],
                                'AvgAgeAutoAccounts': [aa],
                                'TotalAutoAccountsNeverDelinquent': [nd],
                                'WorstDelinquency': [wd],
                                'HasInquiryTelecomm': [it]})

graph = typical_person.copy()

i, j = 0.1, 0
while i <= 1:
    graph[feature_select][j] = panda[feature_select].quantile(i)
    
    i, j = i+0.1, j+1
    
    if i <= 1:
        graph = graph.append(typical_person, ignore_index=True, sort=False)
        
graph = graph.drop_duplicates()
to_predict = helper.pipe(graph, output='x')

discriminator = pickle.load(open('Pickles/discriminator_sm.pkl', 'rb'))
preds = discriminator.predict(to_predict.to_numpy())

cov = discriminator.cov_params()
gradient = (preds * (1 - preds) * to_predict.T).T
std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient.to_numpy()])

st.subheader('Predicted Probabilities')
sea.set_style('whitegrid')
pred_fig, pred_ax = plt.subplots(1, figsize=(7, 4))
sea.scatterplot(x=graph[feature_select], y=preds)
plt.errorbar(x=graph[feature_select], y=preds, yerr=1.96*std_errors, ls='none')
plt.ylabel('Predicted First Year Delinquency Rate')
plt.ylim(0, 1)

st.pyplot(pred_fig)
plt.close()
st.write('')
