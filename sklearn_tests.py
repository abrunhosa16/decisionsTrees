import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from preProcess import PreprocessData
from decisionTreeClassifier import DecisionTreeClassifier
from statistic import Statistics
import itertools

df = pd.read_csv('datasets/restaurant.csv', )

process = PreprocessData(dataset= df)
process.prepare_dataset(func= process.eq_frequency, n_classes= 2)


df = process.dataset
acc = []
dt = DecisionTreeClassifier()

def leave_one_out(df):
    acc = []
    for i in range(len(df)):
        train = pd.concat([df.iloc[:i], df.iloc[i+1:]])
        test = df.iloc[[i]]
        acc.append((train, test))
    return acc
        
