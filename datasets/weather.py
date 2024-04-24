import pandas as pd, numpy as np, matplotlib as plt

df = pd.read_csv('datasets/weather.csv')

df = df.drop('ID', axis=1)

#atributos binários
dic = {"play": {"yes": 1, "no": 0}, "windy": {True:1, False:0}}
df["Play"] = df["Play"].map(dic["play"])
df["Windy"] = df["Windy"].map(dic["windy"])

#Weather
weather_dict = {'rainy': 0, 'overcast': 1, 'sunny': 2}
df['Weather'] = df['Weather'].map(weather_dict)

print(df) 

#falta acabar para variaveis continuas