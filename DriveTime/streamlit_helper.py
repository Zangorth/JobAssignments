from matplotlib import pyplot as plt
from urllib.request import urlopen
import statsmodels.api as sm
import cloudpickle as cp
import seaborn as sea
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import helper


def pipeline_example():
    st.title('DriveTime')
    st.header('Pipeline Example')
    st.write('')
    
    file = st.sidebar.file_uploader('Customer Data File for Prediction', type=['.csv'], accept_multiple_files=False)
    
    if file != None:
        panda = pd.read_csv(file, encoding='utf8')
        
        st.write('')
        st.write('Input')
        st.dataframe(panda)
    
        probs = helper.pipe(panda)
        ranks = helper.pipe(panda, output='rank')
        panda = pd.DataFrame({'probability': probs, 'rank': ranks})
        
        st.write('')
        st.write('Output')    
        st.dataframe(panda)
        
        download_path = st.sidebar.text_input('Download Path:', f'C:/Users/{os.getlogin()}/Downloads')
        download = st.sidebar.button('DOWNLOAD')
        
        if download:
            panda.to_csv(f'{download_path}/{file.name.replace(".csv", "")} probabilities and ranking.csv', encoding='utf-8', index=False)
            

def worst_delinquency():
    st.title('DriveTime')
    st.header('Worst Delinquency')
    st.write('')
    
    url = 'https://drive.google.com/file/d/1iWpQsRaB0CFht_xkhmZqIrYZ9ChRUHno/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    panda = pd.read_csv(url)
    
    panda.loc[panda['TotalInquiries'].isnull(), 'TotalInquiries'] = np.mean(panda['TotalInquiries'])
    
    good_dq = panda.loc[panda['WorstDelinquency'] < 400]
    bad_dq = panda.loc[panda['WorstDelinquency'] >=400]
    
    good_dq['intercept'] = 1
    bad_dq['intercept'] = 1
    
    good_classifier = sm.Logit(good_dq['FirstYearDelinquency'].values, good_dq[['TotalInquiries', 'intercept']]).fit()
    bad_classifier = sm.Logit(bad_dq['FirstYearDelinquency'].values, bad_dq[['TotalInquiries', 'intercept']]).fit()
    
    out = pd.DataFrame({'WorstDelinquency': ['>400', '<400'], 'TotalInquiries_coef': [good_classifier.params[0], bad_classifier.params[0]],
                        'Intercept_coef': [good_classifier.params[1], bad_classifier.params[1]],
                        'TotalInquiries_pval': [round(good_classifier.pvalues[0], 3), round(bad_classifier.params[0], 3)],
                        'Intercept_pval': [round(good_classifier.pvalues[1], 3), round(bad_classifier.pvalues[1], 3)],
                        'observations': [len(good_dq), len(bad_dq)]})
    
    
    values = list(set(panda['WorstDelinquency'].dropna()))
    values.sort()
    
    for val in values:
        print(val)
        dq_set = panda.loc[panda['WorstDelinquency'] == val, ['FirstYearDelinquency', 'TotalInquiries']]
        dq_set['intercept'] = 1
        classifier = sm.Logit(dq_set['FirstYearDelinquency'].values, dq_set[['TotalInquiries', 'intercept']]).fit(maxiter=500)
        
        append = pd.DataFrame({'WorstDelinquency': [int(val)], 'TotalInquiries_coef': [classifier.params[0]],
                    'Intercept_coef': [classifier.params[1]],
                    'TotalInquiries_pval': [round(classifier.pvalues[0], 3)],
                    'Intercept_pval': [round(classifier.pvalues[1], 3)],
                    'observations': len(dq_set)})
            
        out = out.append(append, ignore_index=True, sort=False)
        
        
    dq_set = panda.loc[panda['WorstDelinquency'].isnull(), ['FirstYearDelinquency', 'TotalInquiries']]
    dq_set['intercept'] = 1
    classifier = sm.Logit(dq_set['FirstYearDelinquency'].values, dq_set[['TotalInquiries', 'intercept']]).fit(maxiter=500)
    
    append = pd.DataFrame({'WorstDelinquency': 'missing', 'TotalInquiries_coef': [classifier.params[0]],
                'Intercept_coef': [classifier.params[1]],
                'TotalInquiries_pval': [round(classifier.pvalues[0], 3)],
                'Intercept_pval': [round(classifier.pvalues[1], 3)],
                'observations': len(dq_set)})
        
    out = out.append(append, ignore_index=True, sort=False)
    out['WorstDelinquency'] = out['WorstDelinquency'].astype(str)
    
    st.dataframe(out)
    
    

def plots():
    st.title('DriveTime')
    st.header('Effect Plots')
    
    url = 'https://drive.google.com/file/d/1iWpQsRaB0CFht_xkhmZqIrYZ9ChRUHno/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    panda = pd.read_csv(url)
    
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
    
    discriminator = pickle.load(urlopen('https://drive.google.com/uc?export=download&id=164_xw73fhJFxCQbj6prg3slysxGvxKhP'))
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
    plt.xlim(0, np.max(graph[feature_select]) + np.max(graph[feature_select])*0.05)
    
    st.pyplot(pred_fig)
    plt.close()
    st.write('')
    
    
    
def model():
    st.title('DriveTime')
    st.header('Model Results - ROC AUC 0.6')
    st.write('')
    discriminator = pickle.load(urlopen('https://drive.google.com/uc?export=download&id=164_xw73fhJFxCQbj6prg3slysxGvxKhP'))
    summary = discriminator.summary().tables[1].as_html()
    summary = pd.read_html(summary, header=0, index_col=0)[0]
    st.dataframe(summary)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    