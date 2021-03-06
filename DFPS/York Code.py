import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

# Read in all data
panda = pd.read_excel('ResearchSpecialistV_PCSData.xlsx', skiprows=5)
panda.columns = [column if column != 'Placement County Name' else 'County Name' for column in panda.columns]

facilities = pd.read_excel('ResearchSpecialistV_PCSData.xlsx', skiprows=5, sheet_name=1, skipfooter=6,
                           usecols=['Placement ID', 'Facility Type'])
facilities = facilities.drop_duplicates().reset_index(drop=True)

fips = pd.read_excel('fips.xlsx', skipfooter=2, usecols=['County Name', 'FIPS #'], dtype=str)
fips['FIPS #'] = '48' + fips['FIPS #']

# Merge fips and check
panda['County Name'] = panda['County Name'].str.lower()
fips['County Name'] = fips['County Name'].str.lower()

print(set(panda['County Name']).difference(set(fips['County Name'])))
panda.loc[panda['County Name'] == 'lasalle', 'County Name'] = 'la salle'

panda = panda.merge(fips, how='left', indicator=True, on='County Name')
print(set(panda['_merge']))
panda = panda.drop('_merge', axis=1)

# Merge facilities and check
panda = panda.merge(facilities, how='left', indicator=True, on='Placement ID')
print(set(panda['_merge']))
panda = panda.drop('_merge', axis=1)

# Some facilities appear in multiple counties, so we need a unique county-facility code
cross = pd.crosstab(panda['Placement County Code'], panda['Placement Name'])
panda['facility'] = panda['County Name'] + ' ' + panda['Placement Name']

# Create need based weights, 210 is the modal value, so I'll assign the few missing values to that
panda['ALOC'].value_counts()
panda['weight'] = 1

LOC = 220
for i in range(1, 5):
    panda.loc[panda['ALOC'] == LOC, 'weight'] = 1.15**i
    LOC += 10

# Get the average/weighted average number of children per facility, per county, to assess need
panda['cpf'] = panda.groupby('facility')['facility'].transform('size')
panda['avg_cpf'] = panda.groupby('Placement County Code')['cpf'].transform('mean')

panda['wgt_cpf'] = panda.groupby('facility')['weight'].transform('sum')
panda['wgt_avg_cpf'] = panda.groupby('Placement County Code')['wgt_cpf'].transform('mean')

counties = panda[['County Name', 'FIPS #', 'avg_cpf', 'wgt_avg_cpf']].drop_duplicates().reset_index(drop=True)

# Draw the plots
tex = gpd.read_file('Texas_County_Boundaries_Detailed.shp')
tex = tex.loc[tex['CNTY_FIPS'].isin(counties['FIPS #'])]

tex = tex.set_index('CNTY_FIPS').join(counties.set_index('FIPS #'))

tex.plot(column='avg_cpf', cmap='Reds', edgecolor='black', legend=True)
plt.title('Average Number of Children per Facility')
plt.xticks([])
plt.yticks([])

tex.plot(column='wgt_avg_cpf', cmap='Reds', edgecolor='black', legend=True)
plt.title('Weighted Average Number of Children per Facility')
plt.xticks([])
plt.yticks([])
plt.savefig('wgt_avg_cpf.png')

# Determine how many children are placed in a different region than they originated from
panda['Legal Region'] = panda['Legal Region'].astype(int)
panda['Placement Region'] = panda['Placement Region'].astype(int)
panda['local'] = np.where(panda['Placement Region'] == panda['Legal Region'], 1, 0)
panda['legal_size'] = panda.groupby('Legal Region')['Legal Region'].transform('size')

non_local = panda.loc[panda['local'] == 0]

# Backtracking here, I wanted to check where the children from regions 4 and 9 were going
non_local.loc[non_local['Legal Region'] == 4, 'Placement Region'].value_counts()
non_local.loc[non_local['Legal Region'] == 9, 'Placement Region'].value_counts()

# Examine where they are going, just because it's interesting and maybe relevant
incoming = non_local.groupby('Placement Region').size().sort_index()

# Get percent of children from a legal region that were placed in a different region
non_local['outgoing'] = non_local.groupby('Legal Region')['Legal Region'].transform('size')
non_local = non_local[['Legal Region', 'legal_size', 'outgoing']].drop_duplicates().reset_index(drop=True)
non_local['percent_leaving'] = non_local['outgoing']/non_local['legal_size']
non_local = non_local.sort_values('Legal Region').reset_index(drop=True)

non_local['incoming'] = list(incoming)

region_fips = panda[['Placement Region', 'FIPS #']].drop_duplicates().reset_index(drop=True)
region_fips.columns = ['Legal Region', 'FIPS #']
region_fips = region_fips.merge(non_local, how='left', on='Legal Region')

# Plot the regional graphs
tex = gpd.read_file('Texas_County_Boundaries_Detailed.shp')
tex = tex.loc[tex['CNTY_FIPS'].isin(region_fips['FIPS #'])]
tex = tex.set_index('CNTY_FIPS').join(region_fips.set_index('FIPS #'))

tex.plot(column='percent_leaving', cmap='Reds', edgecolor='black', legend=True)
plt.title('Percent of Children from Region \nPlaced in a Different Region')
plt.xticks([])
plt.yticks([])
plt.savefig('percent leaving.png')

tex.plot(column='incoming', cmap='Reds', edgecolor='black', legend=True)
plt.title('Destination Region of Outgoing Children')
plt.xticks([])
plt.yticks([])
plt.savefig('incoming.png')

tex.plot(column='incoming', cmap='Reds', edgecolor='black')

