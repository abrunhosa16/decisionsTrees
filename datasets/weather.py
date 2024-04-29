import pandas as pd, numpy as np, matplotlib as plt, seaborn as sns

weather_df = pd.read_csv('datasets/weather.csv')

weather_df = weather_df.drop('ID', axis=1)

#atributos binários
dic = {"play": {"yes": 1, "no": 0}, "windy": {True:1, False:0}}
weather_df["Play"] = weather_df["Play"].map(dic["play"])
weather_df["Windy"] = weather_df["Windy"].map(dic["windy"])

#Weather
weather_dict = {'rainy': 0, 'overcast': 1, 'sunny': 2}
weather_df['Weather'] = weather_df['Weather'].map(weather_dict)

 
weather_df['Temp']  = pd.cut(weather_df['Temp'], bins=[50, 64, 100], labels=[0,1])
weather_df['Humidity'] = pd.cut(weather_df['Humidity'], bins=[60, 76, 100], labels=[0,1] )
# no weather.ipynb fiz a avaliação dos quartis Temp e humidity para a seleção dos bins 
# se lê bins=[50, 67, 74, 79, 100] primeiro intervalo de 50 a 67, segundo 67 a 74 ..., o numero de labels precisam ser len(bins) - 1
