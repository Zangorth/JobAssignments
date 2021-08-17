from sqlalchemy import create_engine
import pyodbc as sql
import pandas as pd
import urllib
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\Deloitte')

#############
# Data Read #
#############

males = pd.read_csv('MA_Exer_PikesPeak_Males.txt', encoding='ISO-8859-1', delimiter='\t')
males['gender'] = 1

females = pd.read_csv('MA_Exer_PikesPeak_Females.txt', encoding='ISO-8859-1', delimiter='\t')
females['gender'] = 0

people = males.append(females, ignore_index=True, sort=False)

people.columns = ['place', 'division_place', 'name', 'bib_num', 'age', 'hometown', 'gun_time', 'net_time', 'pace', 'gender']

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
CREATE CLUSTERED INDEX IX_DELOITTE
ON JOBS.deloitte.people(Place, gender)
'''

csr.execute(query)
csr.commit()
con.close()