import pandas as pd, numpy as np, matplotlib as plt

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv('datasets/restaurant.csv')

df = df.drop('ID', axis=1)

df['Pat'] = df['Pat'].fillna('None')

binary = {'Yes':1, 'No':0}
pat_dict = {'None':0, 'Some':1, 'Full':2}
price_dict = {'$':0, '$$':1, '$$$':2}
type_dict = {'French':0, 'Thai':1, 'Burger':2, 'Italian':3}
est_dict = {'0-10':0, '10-30':1, '30-60':2, '>60':3}

df['Pat'] = df['Pat'].map(pat_dict)
df['Price'] = df['Price'].map(price_dict)
df['Type'] = df['Type'].map(type_dict)
df['Est'] = df['Est'].map(est_dict)
df = df.replace(binary)

print(df)