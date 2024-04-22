import pandas as pd, numpy as np, matplotlib as plt

df = pd.read_csv('datasets/restaurant.csv')

df = df.drop('ID', axis=1)

binary_attributes = ['Alt', 'Bar', 'Fri',' Hun', 'Rain', 'Class']
binary = {'Yes':1, 'No':0}
pat_dict = {'None':0, 'Some':1, 'Full':2}
price_dict = {'$':0, '$$':1, '$$$':2}
type_dict = {'French':0, 'Thai':1, 'Burger':2, 'Italian':3}
est_dict = {'0-10':0, '10-30':1, '30-60':2, '>60':3}
df = df.replace(binary)
df = df.replace(pat_dict)
df = df.replace(price_dict)
df = df.replace(type_dict)
df = df.replace(est_dict)

from decisionTree import *

print(entropy(df, binary)) 