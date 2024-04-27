import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

restaurant_df = pd.read_csv('datasets/restaurant.csv')

restaurant_df = restaurant_df.drop('ID', axis=1)

restaurant_df['Pat'] = restaurant_df['Pat'].fillna('None')

binary = {'Yes':1, 'No':0}
dic={
    'Pat': {'None':0, 'Some':1, 'Full':2},
    'Price': {'$':0, '$$':1, '$$$':2},
    'Type': {'French':0, 'Thai':1, 'Burger':2, 'Italian':3},
    'Est': {'0-10':0, '10-30':1, '30-60':2, '>60':3}
}

for feature, mapping in dic.items():
    restaurant_df[feature] = restaurant_df[feature].map(mapping)
    
restaurant_df = restaurant_df.replace(binary)
