import pandas as pd, numpy as np, seaborn, matplotlib.pyplot as plt

'''
    formas de separar var. continuas: equal interval width, equal frequency, k-means
'''

iris_df = pd.read_csv('datasets/iris.csv')

iris_df = iris_df.drop('ID', axis=1)

class_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_df['class'] = iris_df['class'].map(class_dict)

def get_insert_position(l: list, value) -> int: #função para descobrir em qual classe o valor se encaixa
    for idx, i in enumerate(l):
        if value <= i:
            return idx
    return len(l)

#EQUAL FREQUENCY
def eq_frequency(dataset: pd.DataFrame, n_classes: int, feature: str) -> pd.DataFrame:
    split_values = [] #valores onde dar split
    
    #descobre quartis
    for i in range(1, n_classes):
        quantil = (i / n_classes) * 100 
        split_value = np.percentile(iris_df[feature], quantil)
        split_values.append(split_value)
        
    #lista com os valores das novas classes para cada sample
    new_column = [get_insert_position(split_values, row[feature]) for _, row in dataset.iterrows()]
    
    new_dataset = pd.DataFrame({feature + '_discrete': new_column})
    dataset = pd.concat([new_dataset, dataset], axis=1)

    return dataset       

#EQUAL INTERVAL WIDTH
def eq_interval_width(dataset:pd.DataFrame, n_classes: int, feature: str) -> pd.DataFrame:
    min = dataset[feature].min()
    max = dataset[feature].max()
    width = (max - min) / n_classes
    
    split_values = [min + i * width for i in range(1, n_classes)]

    new_column = [get_insert_position(split_values, row[feature]) for _, row in dataset.iterrows()]
    
    new_dataset = pd.DataFrame({feature + '_discrete': new_column})
    dataset = pd.concat([new_dataset, dataset], axis=1)
    
    return dataset

#K-MEANS

def discrete_dataset(dataset: pd.DataFrame, n_classes: int, func) -> pd.DataFrame:
    for feature in dataset.columns:
        if dataset[feature].dtype == float:
            dataset = func(dataset, n_classes, feature)
    return dataset

iris_df = discrete_dataset(iris_df, 4, eq_frequency)
seaborn.pairplot(iris_df, hue='sepallength_discrete', vars=[ 'sepallength',  'sepalwidth',  'petallength',  'petalwidth' ])
plt.show()
