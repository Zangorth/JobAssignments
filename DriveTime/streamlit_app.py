from matplotlib import pyplot as plt
from urllib.request import urlopen
from scipy.stats import rankdata
import statsmodels.api as sm
import streamlit as st
import seaborn as sea
import pandas as pd
import numpy as np
import sklearn
import lxml
import pickle
import re

if 'discriminator_sm' not in st.session_state:
    api = 'AIzaSyAftHJhz8-5UUOACb46YBLchKL78yrXpbw'
    panda = '1iWpQsRaB0CFht_xkhmZqIrYZ9ChRUHno'
    imputer = '1L4gfNu_nTiXYMUHBn-7bGu_VBCgUF63t'
    scaler = '1q29dWx-YLNQwmv2jFO1Db5amEgthtvQK'
    KMeans = '1VPrzk_u9_vzTqDUMHLgqdoN__GhT1QFH'
    discriminator = '1O2Rlm7cSu4LZXtqe5Pj-OIX8j8qbvF-t'
    discriminator_sm = '164_xw73fhJFxCQbj6prg3slysxGvxKhP'
    
    st.session_state['panda'] = pd.read_csv(f'https://www.googleapis.com/drive/v3/files/{panda}?key={api}&alt=media')
    st.session_state['imputer'] = pickle.load(urlopen(f'https://www.googleapis.com/drive/v3/files/{imputer}?key={api}&alt=media'))
    st.session_state['scaler'] = pickle.load(urlopen(f'https://www.googleapis.com/drive/v3/files/{scaler}?key={api}&alt=media'))
    st.session_state['KMeans'] = pickle.load(urlopen(f'https://www.googleapis.com/drive/v3/files/{KMeans}?key={api}&alt=media'))
    st.session_state['discriminator'] = pickle.load(urlopen(f'https://www.googleapis.com/drive/v3/files/{discriminator}?key={api}&alt=media'))
    st.session_state['discriminator_sm'] = pickle.load(urlopen(f'https://www.googleapis.com/drive/v3/files/{discriminator_sm}?key={api}&alt=media'))


############
# Pipeline #
############
def pipe(data, imputer, scaler, km, discriminator=None, output='prob'):
    IV=['AgeOldestIdentityRecord', 'AgeOldestAccount', 'AgeNewestAutoAccount', 'TotalInquiries', 
        'AvgAgeAutoAccounts', 'TotalAutoAccountsNeverDelinquent', 'WorstDelinquency', 'HasInquiryTelecomm']
    
    if type(data) == str:
        panda = pd.read_csv(data, encoding='utf-8')
        
    elif type(data) == pd.core.frame.DataFrame:
        panda = data
    
    if len(set(IV)) > len(set(panda[IV].columns)):
        print('Missing Variables, required Variables include:')
        print(IV)
        return None
    
    if len(set(IV)) < len(panda[IV].columns):
        print('Duplicate Columns Present')
        return None
    
    x = panda[IV].drop('HasInquiryTelecomm', axis=1)
    x['missing'] = x.isnull().sum(axis=1)
    
    x = pd.DataFrame(imputer.transform(x), columns=x.columns)
    x = pd.DataFrame(scaler.transform(x), columns=x.columns)
    
    x['HasInquiryTelecomm'] = np.where(panda['HasInquiryTelecomm'].isnull(), 0, panda['HasInquiryTelecomm'])
        
    km = np.argmax(km.transform(x), axis=1)
    km = pd.get_dummies(km, prefix='km')
    
    for i in range(5):
        if f'km_{i}' not in km.columns:
            km[f'km_{i}'] = 0
            
    km = km[[f'km_{i}' for i in range(len(km.columns))]]
    
    x = x.merge(km, how='left', left_index=True, right_index=True)
    
    if output=='x':
        return x
    
    if output=='prob':
        return discriminator.predict_proba(x)[:, 1]
    
    elif output=='rank':
        return rankdata(-discriminator.predict_proba(x)[:, 1], method='max')


#####################
# Streamlit Helpers #
#####################
def pipeline_example(imputer, scaler, km, discriminator):
    st.title('DriveTime')
    st.header('Pipeline Example')
    
    file = st.sidebar.file_uploader('Customer Data File for Prediction', type=['.csv'], accept_multiple_files=False)
    
    if file == None:
        left, right = st.columns(2)
        
        left.subheader('Assignment')
        comment = '''
        Prepare a scoring pipeline to use your predictive  model(s) in a production environment. 
        To do this you will need to accomplish the following: \n
            * Ingest new loans for scoring \n
            * Perform any required data preprocessing steps such as missing data imputation \n
            * Score the loans using your models \n
            * Output the scores so your business partners can match the scores to the new loans \n
        '''
        
        left.markdown(comment)
        
        download_path = st.sidebar.text_input('Download Path:')
        sample_file = st.sidebar.button('Download Sample File for Testing')
        
        if sample_file:
            sample = st.session_state['panda']
            sample.to_csv(f'{download_path}/sample_file.csv', encoding='utf8', index=False)
        
        
    
    if file != None:
        panda = pd.read_csv(file, encoding='utf8')
        
        st.write('')
        st.write('Input')
        st.dataframe(panda)
    
        probs = pipe(panda, imputer, scaler, km, discriminator)
        ranks = pipe(panda, imputer, scaler, km, discriminator, output='rank')
        panda = pd.DataFrame({'probability': probs, 'rank': ranks})
        
        st.write('')
        st.write('Output')    
        st.dataframe(panda)
        
        download_path = st.sidebar.text_input('Download Path:')
        download = st.sidebar.button('DOWNLOAD')
        
        if download:
            panda.to_csv(f'{download_path}/{file.name.replace(".csv", "")} probabilities and ranking.csv', encoding='utf-8', index=False)
            

def worst_delinquency(panda):
    st.title('DriveTime')
    
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
    
    left, right = st.columns(2)
    
    left.subheader('Assignment')
    
    comment = '''
    Split up the dataset by the WorstDelinquency Variable. Run a regression of First Year delinquency against
    Total Inquiries. Extract the predictor's coefficient and p-value from each model. Combine the results
    in a list where the names of the list correspond to the values of Worst Delinquency'
    '''
    
    left.markdown(comment)
    
    st.subheader('Worst Delinquency')
    st.dataframe(out)
    
    
    
    

def plots(panda, imputer, scaler, km, discriminator):
    st.title('DriveTime')
    
    features = [col for col in panda.columns if col not in ['ID', 'FirstYearDelinquency']]
    features.sort()
    
    st.sidebar.subheader('Features')
    feature_select = st.sidebar.selectbox('', features)
    
    name = re.sub(r"(\w)([A-Z])", r"\1 \2", feature_select)
    
    st.header(f'Effect Plots - {name}')
    
    left, right = st.columns(2)
    
    ##############
    # Univariate #
    ##############
    
    left.subheader('Distribution')
    
    to_plot = panda[feature_select].fillna(panda[feature_select].min()-panda[feature_select].std()*5)
    
    sea.set_style('whitegrid')
    uni_fig, uni_ax = plt.subplots(1, figsize=(7, 4))
    sea.histplot(x=to_plot, stat='percent')
    plt.axvline(panda[feature_select].mean(), ls='--', color='red')
    
    left.pyplot(uni_fig)
    plt.close()
    left.write('')
    
    
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
    
    right.subheader('Bivariate Relationship')
    
    sea.set_style('whitegrid')
    bi_fig, bi_ax = plt.subplots(1, figsize=(7, 4))
    sea.pointplot(x='quantile', y='FirstYearDelinquency', hue='quantile', data=panda)
    plt.ylabel('First Year Delinquency Rate')
    plt.xlabel(name)
    plt.legend().remove()
    
    right.pyplot(bi_fig)
    plt.close()
    right.write('')
    
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
    to_predict = pipe(graph, imputer, scaler, km, output='x')
    
    preds = discriminator.predict(to_predict.to_numpy())
    
    cov = discriminator.cov_params()
    gradient = (preds * (1 - preds) * to_predict.T).T
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient.to_numpy()])
    
    left.subheader('Predicted Probabilities')
    sea.set_style('whitegrid')
    pred_fig, pred_ax = plt.subplots(1, figsize=(7, 4))
    sea.scatterplot(x=graph[feature_select], y=preds)
    plt.errorbar(x=graph[feature_select], y=preds, yerr=1.96*std_errors, ls='none')
    plt.ylabel('Predicted First Year Delinquency Rate')
    plt.ylim(0, 1)
    plt.xlim(0, np.max(graph[feature_select]) + np.max(graph[feature_select])*0.05)
    
    left.pyplot(pred_fig)
    plt.close()
    left.write('')
    
    ############
    # Comments #
    ############
    negative = True if preds[-1] < preds[0] else False
    
    right.subheader('Comments')
    
    comment = f'''
    * Average value for {name} is {round(panda[feature_select].mean(), 2)}
    * {round(panda[feature_select].isnull().mean()*100, 0)}% of the values in this feature are missing
    * {'Negative' if negative else 'Positive'} effect of {name} on First Year Delinquency
    * For the same user, moving from the 10th percentile of {name} to the 90th results in a {round(abs(preds[-1] - preds[0]), 0)*100}% {'increase' if not negative else 'decrease'} in First Year Delinquency Rate
    '''
    
    right.write(comment)
    
    
def model(discriminator):
    st.title('DriveTime')
    
    left, right = st.columns(2)
    
    left.subheader('Model Results - ROC AUC 0.6')
    summary = discriminator.summary().tables[1].as_html()
    summary = pd.read_html(summary, header=0, index_col=0)[0]
    left.dataframe(summary)
    
    right.subheader('Assignment')
    
    comments = '''
    Load the provided data from the accompanying file. Perfrom simple exploratory data analysis, including summary
    statistics and visualizations of the distributions of the data and relationships (see Plots Tab).
    Then build one (or more) model(s) using the provided data. The objective of modeling with this data is 
    to be able to redict the probability that new accounts will become delinquent in the first year 
    Our primary concern is to differentiate low risk accounts from high risk accounts in a rank order fashion.
    '''
    
    right.markdown(comments)
    right.write('')
    
    comments = '''
    Make sure to identify the strongest predictor variables and provide interpretations. 
    Identify and explain any issues or caveats with the data and the models. Calculate predictions
    and show model performance on out of sample data. Finally, summarize results in a fashion that differentiates
    higher risk accounts from lower risk accounts. 
    '''
    
    right.markdown(comments)
    
    
def summary():
    st.title('DriveTime')
    
    options = st.sidebar.radio('', ['Model Results', 'Model Selection'])
    
    if options == 'Model Results':
        st.header('Model Results - Conclusions/Findings')
        
        comment = '''
        * Limited Model efficacy due to poor data quality
            * Multiple features have missing values for close to 50% of observations
            * Missing data imputation helps preserve what trends exist in the data, but is not a substitute for real data
        * Limited model efficacy due to poor feature quality
            * Many of the features seem to target the same idea, which could just be proxies for the same unseen variable (i.e. the age of the customer)
            * All features target credit related factors, while leaving out factors about the customer, dealer, vehicle, and structure of the loan; including such variables should improve performance
        * Overall Findings
            * Low risk individuals tend to have a more well-established credit history (potentially older), with fewer historical delinquencies and fewer telecom inquiries
        '''
        
        st.markdown(comment)
        
    if options == 'Model Selection':
        st.header('Model Selection')
        
        comment = '''
        * Final Model Selected was a Logistic Regression with a Cross Validated ROC AUC of 0.6
            * Feature selection was conducted by examining all possible combinations of variables; normally this is prohibitively timely, but there were only 14 features so in this case it only took 20-30 minutes
            * Logistic regression was designed as the “simple” model, so no interactions or polynomial terms were included; if those are important, they should show up in one of the other methods (e.g. neural networks will automatically detect interactions)
            * The model including all 14 features had the highest (tied for the highest, technically) Cross Validated ROC AUC score, therefore, it was chosen as the best logistic regression
        '''
        
        st.markdown(comment)
        st.write('')
        
        comment = '''
        * ROC AUC
            * ROC AUC was chosen over other possible metrics because it directly indicates how good a model is at ranking predictions, which directly relates to our primary concern of differentiating low risk accounts from high risk accounts.
        '''
        st.markdown(comment)
        st.write('')
        
        comment = '''
        * Other Models Tested
            * Random Forest Classifier
                * ROC AUC: 0.55
                * Optimized over number of estimators
            * Gradient Boosting Classifier
                * ROC AUC: 0.6
                * Same score as logit but much more computationally expensive
                * Optimized over number of estimators, max depth, and learning rate
            * Neural Network
                * ROC AUC: 0.6
                * Same score as logit, but less intuitive and interpretable
                * Optimized over number of layers, neurons, epochs, drop rate, and learning rate
        '''
        st.markdown(comment)
        st.write('')
        
        comment = '''
        * Missing Data Imputation
            * Used simple mean imputation with mode imputation on the categories
            * Also tested KNN Imputation and Iterative Imputation, but neither effected performance
        '''
        st.markdown(comment)
        st.write('')
        
        comment = '''
        * Other Notes
            * Implemented standard scaling to normalize data; other scaling methods could have been tested to see if they impacted performance, but were not
            * Kmeans clusters were included, other clustering methods should be tested to see if they improve performance
        '''
        st.markdown(comment)
        st.write('')


#################
# Streamlit App #
#################

st.set_page_config(layout='wide')
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('', ['Model', 'Plots', 'Summary', 'Worst Delinquency', 'Pipeline'])

if page == 'Model':
    model(st.session_state['discriminator_sm'])

elif page == 'Plots':
    plots(st.session_state['panda'], st.session_state['imputer'], 
          st.session_state['scaler'], st.session_state['KMeans'],
          st.session_state['discriminator_sm'])
    
elif page == 'Summary':
    summary()

elif page == 'Worst Delinquency':
    worst_delinquency(st.session_state['panda'])
    
elif page == 'Pipeline':
    pipeline_example(st.session_state['imputer'], st.session_state['scaler'],
                     st.session_state['KMeans'], st.session_state['discriminator'])

