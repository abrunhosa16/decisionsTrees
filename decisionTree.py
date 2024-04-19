import pandas as pd, numpy as np, sklearn, matplotlib as plt
from restaurant import df

def entropy(df:pd.DataFrame, atribute:str):
    length = df.shape[0]
    atribute_values = df[atribute].value_counts().to_dict()
    h = 0
    for value, times in atribute_values.items():
        prob = times / length
        h -= prob * np.log2(prob)
    return h 
    
print(entropy(df, 'Fri'))