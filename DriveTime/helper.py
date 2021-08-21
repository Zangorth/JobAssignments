from sklearn.model_selection import train_test_split as split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import OrderedDict
from xgboost import XGBClassifier
from scipy.stats import rankdata
from torch import nn
import pandas as pd
import numpy as np
import warnings
import pickle
import torch

device = torch.device('cuda:0')
warnings.filterwarnings('error', category=ConvergenceWarning)

class ParameterError(Exception):
    pass

###########################
# Neural Network Function #
###########################
class Discriminator():
    def __init__(self, shape, drop, transforms, lr_nn, epochs, output, layers=3):
        self.shape, self.drop = shape, drop
        self.output, self.layers = output, layers
        self.transforms = transforms
        self.lr_nn, self.epochs = lr_nn, epochs
        
        return None
        
    class Classifier(nn.Module):
        def __init__(self, shape, transforms, drop, output, layers=3):
            super().__init__()
            
            transforms = [shape] + transforms
            sequential = OrderedDict()
            
            i = 0
            while i < layers:
                sequential[f'linear_{i}'] = nn.Linear(transforms[i], transforms[i+1])
                sequential[f'relu_{i}'] = nn.ReLU()
                sequential[f'drop_{i}'] = nn.Dropout(drop)
                i+=1
                
            sequential['linear_final'] = nn.Linear(transforms[i], output)
            sequential['softmax'] = nn.Softmax(dim=1)
            
            self.model = nn.Sequential(sequential)
            
        def forward(self, x):
            output = self.model(x)
            return output
    
    def fit(self, x, y):
        col_count = x.shape[1]
        x, y = torch.from_numpy(x.values).to(device), torch.from_numpy(y.values).to(device)
        
        train_set = [(x[i].to(device), y[i].to(device)) for i in range(len(y))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**10, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = self.Classifier(col_count, self.transforms, self.drop, self.output, self.layers).to(device)
        optim = torch.optim.Adam(discriminator.parameters(), lr=self.lr_nn)
    
        for epoch in range(self.epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                discriminator.zero_grad()
                yhat = discriminator(inputs.float())
                loss = loss_function(yhat, targets.long())
                loss.backward()
                optim.step()
                
        self.model = discriminator
        
        return None
    
    def predict(self, x):
        discriminator = self.model
        discriminator.to(device).eval()
        
        x = torch.from_numpy(x.values).to(device)
        preds = np.argmax(discriminator(x.float()).cpu().detach(), axis=1)
        
        return preds
    
    def predict_proba(self, x):
        discriminator = self.model
        discriminator.to(device).eval()
        
        x = torch.from_numpy(x.values).to(device)
        preds = discriminator(x.float()).cpu().detach()
        
        return preds
    
###################
# Cross Validater #
###################
def cv(model, x, y, 
       n_samples=100, n_estimators=100, lr_gbc=0.001, max_depth=3,
       transforms=[128, 64, 32, 16], drop=0.1, layers=3, 
       epochs=20, lr_nn=0.0001, output=9, 
       cv=20, frac=0.1, over=False):
    
    auc = []
    
    for i in range(cv):
        x_train, x_test, y_train, y_test = split(x, y, test_size=frac, stratify=y)
        
        if over:
            oversample = SMOTE(n_jobs=-1)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

        if model == 'logit':
            discriminator = LogisticRegression(max_iter=500, fit_intercept=False)

        elif model == 'rfc':
            discriminator = RandomForestClassifier(n_estimators=n_samples, n_jobs=-1)

        elif model == 'gbc':
            discriminator = XGBClassifier(n_estimators=n_estimators, learning_rate=lr_gbc, 
                                          max_depth=max_depth, n_jobs=4, use_label_encoder=False,
                                          eval_metric='mlogloss', tree_method='gpu_hist')

        elif model == 'nn':
            discriminator = Discriminator(x.shape[1], drop, transforms, lr_nn, epochs, output, layers)

        try:
            discriminator.fit(x_train, y_train)
            predictions = discriminator.predict_proba(x_test)[:, 1]
            auc.append(roc_auc_score(y_test, predictions))
        except ConvergenceWarning:
            auc.append(0)
            
    return np.mean(auc)
    
    
################
# Logit Helper #
################
def startup(features, model, x, y):
    x = x[list(features)]
    
    f1 = cv(model, x, y)
    
    return [list(features), len(list(features)), f1]


############
# Pipeline #
############
def pipe(data, output='prob'):
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
    
    imputer = pickle.load(open('Pickles\\imputer.pkl', 'rb'))
    x = pd.DataFrame(imputer.transform(x), columns=x.columns)
    
    scaler = pickle.load(open('Pickles\\scaler.pkl', 'rb'))
    x = pd.DataFrame(scaler.transform(x), columns=x.columns)
    
    x['HasInquiryTelecomm'] = np.where(panda['HasInquiryTelecomm'].isnull(), 0, panda['HasInquiryTelecomm'])
    
    km = pickle.load(open('Pickles\\KMeans.pkl', 'rb'))
    km = np.argmax(km.transform(x), axis=1)
    km = pd.get_dummies(km, prefix='km')
    
    for i in range(5):
        if f'km_{i}' not in km.columns:
            km[f'km_{i}'] = 0
            
    km = km[[f'km_{i}' for i in range(len(km.columns))]]
    
    x = x.merge(km, how='left', left_index=True, right_index=True)
    
    if output=='x':
        return x
    
    discriminator = pickle.load(open('Pickles\\discriminator.pkl', 'rb'))
    
    if output=='prob':
        return discriminator.predict_proba(x)[:, 1]
    
    elif output=='rank':
        return rankdata(-discriminator.predict_proba(x)[:, 1], method='max')
    
    
    
    
    
        
    
        
        
    
    
        

