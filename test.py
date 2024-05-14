import pandas as pd, numpy as np, math, itertools, matplotlib, matplotlib.pyplot as plt, networkx as nx, graphviz
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData
from statistic import Statistics
from node import Node

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')



process = PreprocessData(dataset= restaurant_df)
process.prepare_dataset(n_classes= 3, func= process.eq_interval_width)
process.stratify(0.3)
dt = DecisionTreeClassifier()
dt.fit(process= process)
dt.print_tree()
# stats = Statistics(tree= dt, process= process)
# print('Train:')
# stats.evaluate(process.train)
# print('Test:')
# stats.evaluate(process.test)

import networkx as nx
import matplotlib.pyplot as plt

def add_nodes_with_labels(G, node: Node):
    print(node.children)
    for child in node.children:
        if child.is_leaf():
            child_value = node.feature + str(child.condition) + ': ' + str(child.leaf_value)
        else:
            child_value = child.feature
        G.add_edge(node.feature, child_value, label= child.condition)
        add_nodes_with_labels(G, child)

# Create an empty graph
G = nx.Graph()

# Add nodes to form a complete binary tree with 3 levels (including the root) and labels representing operations
add_nodes_with_labels(G, dt.root)

# Draw the tree with edge labels
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=12, font_weight="bold")
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Complete Binary Tree with Edge Labels")
plt.show()