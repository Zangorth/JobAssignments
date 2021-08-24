from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from multiprocessing import Pool
from functools import partial
import statsmodels.api as sm
from skopt import plots
import seaborn as sea
import pandas as pd
import numpy as np
import pickle
import skopt
import sys
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import helper

panda = pd.read_csv('account-defaults.csv', encoding='utf-8')

################
# Data Munging #
################
y = panda['FirstYearDelinquency']
x = panda.drop(['ID', 'FirstYearDelinquency', 'HasInquiryTelecomm'], axis=1)
x['missing'] = x.isnull().sum(axis=1)

# Impute missing values, using simple imputer to save time
impute = SimpleImputer(strategy='mean').fit(x)
pickle.dump(impute, open('Pickles\\imputer.pkl', 'wb'))
x = pd.DataFrame(impute.transform(x), columns=x.columns)

# Try out different imputation methods to see if we can improve performance
#impute = KNNImputer()
#x = pd.DataFrame(impute.fit_transform(x), columns=x.columns)
#print(f"Selected Columns: {helper.cv('logit', x, y, cv=100)}")

#impute = IterativeImputer(max_iter=500)
#x = pd.DataFrame(impute.fit_transform(x), columns=x.columns)
#print(f"Selected Columns: {helper.cv('logit', x, y, cv=100)}")

scaler = StandardScaler().fit(x)
pickle.dump(scaler, open('Pickles\\scaler.pkl', 'wb'))
x = pd.DataFrame(scaler.transform(x), columns=x.columns)

x['HasInquiryTelecomm'] = np.where(panda['HasInquiryTelecomm'].isnull(), 0, panda['HasInquiryTelecomm'])

# Generate some clusters
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(x)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

km = KMeans(5).fit(x)
pickle.dump(km, open('Pickles\\KMeans.pkl', 'wb'))
km = km.transform(x)
km = np.argmax(km, axis=1)
km = pd.get_dummies(km, prefix='km')

x = x.merge(km, how='left', left_index=True, right_index=True)

##########
# Models #
##########
# Baseline
auc = []
for i in range(20):
    x_train, x_test, y_train, y_test = split(x, y, test_size=0.1, stratify=y)
    
    dumb = DummyClassifier(strategy='uniform')
    dumb.fit(x_train, y_train)
    predictions = dumb.predict_proba(x_test)[:, 1]
    auc.append(roc_auc_score(y_test, predictions))

print(f'Baseline: {np.mean(auc)}')

# Logit
feature_set = x.columns
features = []

for i in range(1, len(feature_set)):
    features += combinations(feature_set, i)

fit = partial(helper.startup, model='logit', x=x, y=y)

with Pool(20) as pool:
    out = pool.map(fit, features)
    pool.close()
    pool.join()
        
fit_frame = pd.DataFrame({'features': [out[i][0] for i in range(len(out))],
                          'size': [out[i][1] for i in range(len(out))],
                          'auc': [out[i][2] for i in range(len(out))]})

    
fit_frame.to_csv('Logit Fit.csv', encoding='utf-8', index=False)
    
sea.set(rc={'figure.figsize':(16, 10)})
sea.set_style('whitegrid')
sea.scatterplot(x=fit_frame['size'], y=fit_frame['auc'])
plt.title('Cross Validated AUC Score by Number of Features')
plt.savefig('Plots\\AUC Logit.png', bbox_inches='tight')
plt.show()
plt.close()

selected = fit_frame.loc[fit_frame.auc == fit_frame.auc.max(), 'features'].item()


helper.cv('logit', x[selected], y)

# All columns perform almost identically to all columns - Total Accounts never Delinquent, so 
# I'm just going to leave them all in for simplicity
print(f"All Columns: {helper.cv('logit', x, y, cv=100)}")
print(f"Selected Columns: {helper.cv('logit', x[selected], y, cv=100)}")


# Random Forest - AUC: 0.55 | {'n_samples': 879}
space = [skopt.space.Integer(50, 1000, name='n_samples')]

i = 0
@skopt.utils.use_named_args(space)
def rfc(n_samples):
    auc = helper.cv('rfc', x, y, n_samples)
        
    global i
    i += 1
    
    print(f'({i}/{100}): {round(auc, 3)}')
    return (- 1.0 * auc)
        
result = skopt.forest_minimize(rfc, space, acq_func='PI', n_jobs=10, n_initial_points=50)

sea.set(rc={'figure.figsize':(16, 10)})
sea.set_style('whitegrid')
sea.scatterplot(x=[result.x_iters[i][0] for i in range(len(result.x_iters))], y=result.func_vals)
plt.title('Random Forest Performance')
plt.savefig('AUC Random Forest.png', bbox_inches='tight')
plt.show()
plt.close()


# Gradient Boosting Classifier - AUC 0.615 | {'n_estimators': 927, 'max_dept': 1, 'lr_gbc': 0.0226}
space = [skopt.space.Integer(50, 1000, name='n_estimators'),
         skopt.space.Integer(1, 10, name='max_depth'),
         skopt.space.Real(0.00001, 0.05, name='lr_gbc', prior='log-uniform')]

i = 0
@skopt.utils.use_named_args(space)
def gbc(n_estimators, max_depth, lr_gbc):
    auc = helper.cv('gbc', x, y, n_estimators=n_estimators, lr_gbc=lr_gbc, max_depth=max_depth)
        
    global i
    i += 1
    
    print(f'({i}/{200}): {round(auc, 3)}')
    return (- 1.0 * auc)
        
result = skopt.forest_minimize(gbc, space, acq_func='PI', n_calls=200, n_initial_points=50)

plots.plot_objective(result)
plots.plot_evaluations(result)


# Nueral Network - AUC: 0.61 | {'layers': 1, 'epochs': 27, 'drop': 0.053928, 'lr_nn'=0.0019949, 'transform': [709, 722, 208, 347, 999]}
space = [
    skopt.space.Integer(1, 5, name='layers'),
    skopt.space.Integer(1, 100, name='epochs'),
    skopt.space.Real(0.0001, 0.2, name='drop', prior='log-uniform'),
    skopt.space.Real(0.00001, 0.04, name='lr_nn', prior='log-uniform')
    ]

space = space + [skopt.space.Integer(2**2, 2**10, name=f'transform_{i}') for i in range(5)]

i = 0
@skopt.utils.use_named_args(space)
def gbc(drop, layers, epochs, lr_nn, **kwargs):
    transforms = [kwargs[key] for key in kwargs if 'transform' in key]
    
    auc = helper.cv('nn', x, y, transforms=transforms, drop=drop, layers=layers, epochs=epochs,
                    lr_nn=lr_nn, output=2)
    
    global i
    i += 1
    
    print(f'({i}/{200}): {round(auc, 3)}')
    return (- 1.0 * auc)
        
result = skopt.forest_minimize(gbc, space, acq_func='PI', n_calls=200, n_initial_points=50)

plots.plot_evaluations(result)


# Comparison
print(f"Logit: {helper.cv('logit', x, y, cv=100)}")
print(f"RFC: {helper.cv('rfc', x, y, cv=100, n_samples=879)}")
print(f"GBC: {helper.cv('gbc', x, y, cv=100, n_estimators=927, max_depth=1, lr_gbc=0.022638)}")
print(f"NN: {helper.cv('nn', x, y, transforms=[709, 722, 208, 347, 999], drop=0.053928, epochs=27, layers=1, lr_nn=0.0019949, cv=100, output=2)}")

discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
discriminator.fit(x, y)

pickle.dump(discriminator, open('Pickles\\discriminator.pkl', 'wb'))

discriminator.predict_proba(x)[:, 1]

discriminator = sm.Logit(y, x).fit(max_iter=500)
pickle.dump(discriminator, open('Pickles\\discriminator_sm.pkl', 'wb'))

print('')
print('')
print(discriminator.summary())




