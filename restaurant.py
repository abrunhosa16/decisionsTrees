import pandas as pd, numpy as np, sklearn, matplotlib as plt
import sklearn.preprocessing

df = pd.read_csv('datasets/restaurant.csv')

df = df.drop('ID', axis=1)

# mudança de variaveis categoricas binarias para binarias
label_encoder = sklearn.preprocessing.LabelBinarizer()
for column in df.columns:
    if len(set(df[column].values)) == 2:
        df[column] = label_encoder.fit_transform(df[column])

for column in df.columns:
    if len(set(df[column].values)) > 2: #atributos com mais do que 2 valores
        label_encoder = sklearn.preprocessing.LabelEncoder() #inicia encoder
        df[column] = label_encoder.fit_transform(df[column]) #cria nova coluna com os valores codificados
        
# print(df.head())