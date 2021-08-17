from sqlalchemy import create_engine
import statsmodels.api as sm
import pyodbc as sql
import pandas as pd
import numpy as np
import urllib
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\Deloitte')

####################
# Helper Functions #
####################
def extract_number(string):
    string = [i for i in [char for char in string] if i.isdigit()]
    return int(''.join(string))

def zero_add(column):
    return pd.Series(np.where(column.str.len() == 1, '0' + column, column))

#############
# Data Read #
#############
males = pd.read_csv('MA_Exer_PikesPeak_Males.txt', encoding='ISO-8859-1', delimiter='\t')
males['gender'] = 'Male'

females = pd.read_csv('MA_Exer_PikesPeak_Females.txt', encoding='ISO-8859-1', delimiter='\t')
females['gender'] = 'Female'

people = males.append(females, ignore_index=True, sort=False)

people.columns = ['place', 'division_place', 'bib_num', 'name', 'age', 'hometown', 'gun_time', 'net_time', 'pace', 'gender']

##############
# Data Clean #
##############
# Net Time
time = people['net_time'].str.split(':')
seconds = [extract_number(t[-1]) for t in time]
minutes = [extract_number(t[-2]) for t in time]
hours = [extract_number(t[-3]) if len(t) > 2 else 0 for t in time]

time = pd.DataFrame({'hours': hours, 'minutes': minutes, 'seconds': seconds})

people['net_time_minutes'] = time.hours*60 + time.minutes + time.seconds/60

time = time.astype(str)
people['net_time'] = zero_add(time.hours) + ':' + zero_add(time.minutes) + ':' + zero_add(time.seconds)

# Gun Time
time = people['gun_time'].str.split(':')
seconds = [extract_number(t[-1]) for t in time]
minutes = [extract_number(t[-2]) for t in time]
hours = [extract_number(t[-3]) if len(t) > 2 else 0 for t in time]

time = pd.DataFrame({'hours': hours, 'minutes': minutes, 'seconds': seconds})

people['gun_time_minutes'] = time.hours*60 + time.minutes + time.seconds/60

time = time.astype(str)
people['gun_time'] = zero_add(time.hours) + ':' + zero_add(time.minutes) + ':' + zero_add(time.seconds)

# Recode Divisions
people['division'] = '00-14'
people.loc[people.age > 14, 'division'] = '15-19'

for i in range(20, 90, 10):
    people.loc[people.age > i, 'division'] = f'{i}-{i+9}'


people = people[['gender', 'place', 'division', 'name', 'bib_num', 'age', 'hometown', 
                 'gun_time', 'gun_time_minutes', 'net_time', 'net_time_minutes',
                 'pace']]

# print quantile for chris doe reference
people.loc[(people.division == '40-49') & (people.gender == 'Male'), 'net_time_minutes'].quantile(0.48)

###############
# Quick Model #
###############

lm = sm.OLS(people.net_time_minutes.values.ravel(), pd.get_dummies(people.division))
res = lm.fit()
print(res.summary())

###############
# Data Upload #
###############
conn_str = (
    r'Driver={SQL Server};'
    r'Server=ZANGORTH\HOMEBASE;'
    r'Database=JOBS;'
    r'Trusted_Connection=yes;'
)
con = urllib.parse.quote_plus(conn_str)

engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')

people.to_sql(name='people', con=engine, schema='deloitte', if_exists='replace', index=False)

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                 
csr = con.cursor()
query = '''
ALTER TABLE JOBS.deloitte.people
ALTER COLUMN gender VARCHAR(7);

CREATE CLUSTERED INDEX IX_DELOITTE
ON JOBS.deloitte.people(Place, gender);
'''

csr.execute(query)
csr.commit()
con.close()