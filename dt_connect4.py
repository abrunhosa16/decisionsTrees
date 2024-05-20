import pickle
import pandas as pd
from decisionTreeClassifier import DecisionTreeClassifier
from board import Board
from preProcess import PreprocessData
from statistic import Statistics
from board import Board
from connect4 import *

data = pd.read_csv('datasets\connect4.csv')

indice = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 
 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 
 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 
 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'Class']

data.columns = indice

connect_df = data

p = PreprocessData(connect_df)
p.stratify(0.1)

dt = DecisionTreeClassifier()
dt.fit(p, dt.max_info_gain)
dt.print_tree()

with open('variables/models.pkl', 'wb') as f:
    pickle.dump([p, dt], f)