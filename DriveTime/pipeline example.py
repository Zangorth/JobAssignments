import pandas as pd
import sys
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Jobs\DriveTime')
import helper

probs = helper.pipe('account-defaults.csv')
ranks = helper.pipe('account-defaults.csv', output='rank')

panda = pd.DataFrame({'probability': probs, 'rank': ranks})