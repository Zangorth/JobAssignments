import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')

st.set_page_config(layout='wide')
st.title('DriveTime')
st.header('Worst Delinquency')

panda = pd.read_csv('account-defaults.csv', encoding='utf-8')
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