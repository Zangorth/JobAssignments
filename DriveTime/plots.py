from matplotlib import pyplot as plt
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

##############
# Univariate #
##############
for col in [col for col in panda.columns if col != 'ID']:
    to_plot = panda[col].fillna(panda[col].min()-panda[col].std()*5)
    
    name = re.sub(r"(\w)([A-Z])", r"\1 \2", col)
    
    sea.set(rc={'figure.figsize':(10.2, 11.34)})
    sea.set_style('whitegrid')
    sea.histplot(x=to_plot, stat='percent')
    plt.title(name, fontdict = {'fontsize' : 24})
    plt.axvline(panda[col].mean(), ls='--', color='red')
    plt.savefig(f'Plots\\{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

#############
# Bivariate #
#############
panda['missing'] = panda.isnull().sum(axis=1)

for col in [col for col in panda.columns if col not in ['ID', 'FirstYearDelinquency']]:
    panda['quantile'] = np.nan
    for i in np.arange(0.1, 1.1, 0.1):
        mn, mx = panda[col].quantile(i-0.1), panda[col].quantile(i)
        
        if mn != mx:
            panda.loc[(panda[col] >= mn) & (panda[col] < mx), 'quantile'] = round(panda.loc[(panda[col] >= mn) & (panda[col] < mx), col].mean(), 2)
        
        else:
            panda.loc[panda[col] == mn, 'quantile'] = round(panda.loc[panda[col] == mn, 'quantile'].mean(), 2)
    
    name = re.sub(r"(\w)([A-Z])", r"\1 \2", col)
    
    sea.set(rc={'figure.figsize':(10.2, 11.34)})
    sea.set_style('whitegrid')
    sea.pointplot(x='quantile', y='FirstYearDelinquency', hue='quantile', data=panda)
    plt.ylabel('First Year Delinquency Rate')
    plt.xlabel(name)
    plt.title(f'Average First Year Delinquency Rate\nby {name}', {'fontsize' : 24})
    plt.legend().remove()
    plt.savefig(f'Plots\\{name} Delinquency.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
name = re.sub(r"(\w)([A-Z])", r"\1 \2", 'HasInquiryTelecomm')
    
sea.set(rc={'figure.figsize':(10.2, 11.34)})
sea.set_style('whitegrid')
sea.pointplot(x='HasInquiryTelecomm', y='FirstYearDelinquency', hue='HasInquiryTelecomm', data=panda)
plt.ylabel('First Year Delinquency Rate')
plt.xlabel(name)
plt.title(f'Average First Year Delinquency Rate\nby {name}', {'fontsize' : 24})
plt.legend().remove()
plt.savefig(f'Plots\\{name} Delinquency.png', bbox_inches='tight')
plt.show()
plt.close()
    
panda = panda.drop('quantile', axis=1)


###########################
# Predicted Probabilities #
###########################

typical_person = pd.DataFrame({'AgeOldestIdentityRecord': panda['AgeOldestIdentityRecord'].mean(),
                               'AgeOldestAccount': panda['AgeOldestAccount'].mean(),
                               'AgeNewestAutoAccount': panda['AgeNewestAutoAccount'].mean(),
                               'TotalInquiries': panda['TotalInquiries'].mean(),
                               'AvgAgeAutoAccounts': panda['AvgAgeAutoAccounts'].mean(),
                               'TotalAutoAccountsNeverDelinquent': panda['TotalAutoAccountsNeverDelinquent'].mean(),
                               'WorstDelinquency': [0],
                               'HasInquiryTelecomm': [0]})

for col in [col for col in typical_person]:
    graph = typical_person.copy()
    
    i, j = 0.1, 0
    while i <= 1:
        graph[col][j] = panda[col].quantile(i)
        
        j += 1
        i += 0.1
        
        if i <= 1:
            graph = graph.append(typical_person, ignore_index=True, sort=False)
            
    graph = graph.drop_duplicates()
    
    to_predict = helper.pipe(graph, output='x')
    
    name = re.sub(r"(\w)([A-Z])", r"\1 \2", col)
    
    discriminator = pickle.load(open('discriminator_sm.pkl', 'rb'))
    preds = discriminator.predict(to_predict.to_numpy())
    
    cov = discriminator.cov_params()
    gradient = (preds * (1 - preds) * to_predict.T).T
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient.to_numpy()])
    
    sea.set(rc={'figure.figsize':(10.2, 11.34)})
    sea.set_style('whitegrid')
    sea.scatterplot(x=graph[col], y=preds)
    plt.errorbar(x=graph[col], y=preds, yerr=1.96*std_errors, ls='none')
    plt.ylabel('Predicted First Year Delinquency Rate')
    plt.title(f'{name}\nPredicted Probabilities', fontdict = {'fontsize' : 24})
    plt.savefig(f'Plots\\{name} Predicted.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    