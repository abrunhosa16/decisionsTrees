import pandas as pd, numpy as np, sklearn, matplotlib as plt
import sklearn.preprocessing

df = pd.read_csv('datasets/restaurant.csv')

df = df.drop('ID', axis=1)

#mudança de variaveis categoricas binarias para binarias
label_encoder = sklearn.preprocessing.LabelBinarizer()
for column in df.columns:
    if len(set(df[column].values)) == 2:
        df[column] = label_encoder.fit_transform(df[column])

df.loc[pd.isnull(df['Pat']), 'Pat'] = 'None' #mudança do NaN para uma str

for column in df.columns:
    if len(set(df[column].values)) > 2: #atributos com mais do que 2 valores
        label_encoder = sklearn.preprocessing.LabelEncoder() #inicia encoder
        df[column + '_num'] = label_encoder.fit_transform(df[column]) #cria nova coluna com os valores codificados

        #para guardar num dict o numero associado ao valor real
        to_num = {}
        for value in set(df[column].values.tolist()):
            num = df.loc[df[column] == value, column + '_num'].to_list()[0]
            to_num[num] = value

        one_hot_encoder = sklearn.preprocessing.OneHotEncoder() #inicia encoder
        num_reshaped = df[column + '_num'].values.reshape(-1, 1) #da reshape para matriz
        one_hot_encoder.fit(num_reshaped) #fit ao encoder
        ohe = one_hot_encoder.transform(num_reshaped)
        ohe_array = ohe.toarray()
        ohe_df = pd.DataFrame(ohe_array, columns=[f'{to_num.get(i)}' for i in range(ohe_array.shape[1])]) #novo dataframe
        df = pd.concat([df, ohe_df], axis=1) #junta o novo df

        df = df.drop([column, column + '_num'], axis=1)
    
print(df)