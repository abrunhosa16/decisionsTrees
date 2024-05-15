import pandas as pd, numpy as np, math, itertools
from node import Node
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData
from statistic import Statistics

'''
    Ideia: mudar o children para um dicionario assim no nó pai temos no dicionario condição: subtree
    Assim escusamos de usar o atributo condition e parece ser mais fácil na parte do predict

    PERGUNTAR SE É PREFERIVEL TER PREVISAO MESMO QUE MAIS ERRADA OU NAO TER
'''

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')


def run(dataset: int, n_classes: int, f: int):
    datasets = [restaurant_df, weather_df, iris_df]
    dataset = datasets[dataset - 1]
    process = PreprocessData(dataset= dataset)
    funcs = [process.eq_frequency, process.eq_interval_width, process.k_means]
    f = funcs[f - 1]
    process.prepare_dataset(n_classes= n_classes, func= f)
    process.stratify(0.2)
    print(process.train)
    dt = DecisionTreeClassifier()
    dt.fit(process)
    dt.print_tree()
    
# preProcess = PreprocessData(restaurant_df)
# preProcess.prepare_dataset(n_classes= 3, func= preProcess.eq_frequency)
# train, test = preProcess.stratify(0.2)
# dt = DecisionTreeClassifier()
# train_df, test_df = dt.stratify(preProcess.dataset, 0.2)
# dt.fit(preProcess.dataset)
# dt.print_tree()
# dt.evaluate(test_dataset= test_df)

# for feature in preProcess.dataset.columns[:-1]:
#     print(dt.info_gain(dt.original_dataset, feature))
