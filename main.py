import pandas as pd, numpy as np, math, itertools
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData
from statistic import Statistics

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')

while True:
    try:
        dataset = int(input('Qual dataset quer usar? 1- Restaurante  2- Tempo  3- Iris: '))
        if dataset in {1,2,3}:
            break
        else:
            print('Insira um inteiro entre 1 e 3.')
    except ValueError:
        print('Insira um inteiro entre 1 e 3.')
        
while True:
    try:
        n_classes = int(input('Quantas classes quer usar para separar as features contínuas? 2-5: '))
        if n_classes in range(2,6):
            break
        else:
            print('Insira um inteiro entre 2 e 5.')
    except ValueError:
        print('Insira um inteiro entre 2 e 5.')
        
while True:
    try:
        f = int(input('Qual função quer usar para fazer essa separação? 1- Eq. Frequency  2- Eq. Int. Width: '))
        if f in {1,2}:
            break
        else: 
            print('Insira 1 ou 2.')
    except ValueError:
        print('Insira 1 ou 2.')
while True:
    try:
        gain = int(input('Qual método quer usar? 1- Info Gain  2- Gini: '))
        if gain in {1,2}:
            break
        else:
            print('Insira 1 ou 2.')
    except ValueError:
        print('Insira 1 ou 2.')
while True:
    try:
        eval = int(input('Quer gerar uma árvore apenas ou várias e ver a média das suas estatísticas? '))
        if isinstance(eval, int):
            break
    except ValueError:
        print('Insira um inteiro.')
    

if eval == 1:
    datasets = [restaurant_df, weather_df, iris_df]
    
    process = PreprocessData(dataset= datasets[dataset - 1])
    funcs = [process.eq_frequency, process.eq_interval_width]
    process.prepare_dataset(n_classes= n_classes, func= funcs[f - 1])
    process.stratify(0.2)

    dt = DecisionTreeClassifier()
    gains = [dt.max_info_gain, dt.max_gini]
    dt.fit(dataset= process, option= gains[ gain - 1 ])
    dt.print_tree()
else:
    stats = Statistics()
    print((stats.evaluate_n_times(dataset= dataset, n= eval, n_classes= n_classes, f= f, gain= gain)))