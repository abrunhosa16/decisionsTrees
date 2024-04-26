import pandas as pd, numpy as np, matplotlib as plt, seaborn as sns

df = pd.read_csv('datasets/weather.csv')

df = df.drop('ID', axis=1)

#atributos binários
dic = {"play": {"yes": 1, "no": 0}, "windy": {True:1, False:0}}
df["Play"] = df["Play"].map(dic["play"])
df["Windy"] = df["Windy"].map(dic["windy"])

#Weather
weather_dict = {'rainy': 0, 'overcast': 1, 'sunny': 2}
df['Weather'] = df['Weather'].map(weather_dict)


df['Temp']  = pd.cut(df['Temp'], bins=[50, 67, 74, 79, 100], labels=[0,1,2, 3])
df['Humidity'] = pd.cut(df['Humidity'], bins=[60, 71, 83, 90, 100], labels=[0,1,2, 3] )
print(df)
weather1 = df
# no weather.ipynb fiz a avaliação dos quartis Temp e humidity para a seleção dos bins 
# se lê bins=[50, 67, 74, 79, 100] primeiro intervalo de 50 a 67, segundo 67 a 74 ..., o numero de labels precisam ser len(bins) - 1

