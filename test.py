import pandas as pd, numpy as np, math, itertools, matplotlib, matplotlib.pyplot as plt, networkx as nx, graphviz
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData
from statistic import Statistics
from node import Node
import graphviz

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')


def visualize_tree(tree: Node, target_name: str, label_names):
    leaf_names = label_names[target_name]
    def traverse(node: Node, graph, parent_name=None, edge_label=""):
        node_id = str(id(node))
        if node.is_leaf():
            node_label = f"Class: {[chave for chave, valor in leaf_names.items() if valor == node.leaf_value][0]}"
            node_color = 'lightblue'

        else:
            node_label = f"{node.feature}\nIG: {node.info_gain:.2f}"
            node_color = 'red'


        graph.node(node_id, label=node_label, style='filled', fillcolor=node_color, shape='box')
        if parent_name:
            graph.edge(parent_name, node_id, label=edge_label, color='black')

        for child in node.children:
            traverse(child, graph, node_id, f"= {child.condition}")

    dot = graphviz.Digraph()
    traverse(tree, dot)
    return dot


process = PreprocessData(dataset= restaurant_df)
process.prepare_dataset(n_classes= 3, func= process.eq_interval_width)

process.stratify(0.2)
dt = DecisionTreeClassifier()
dt.target = 'class'
dt.fit(dataset= process.dataset, option= dt.max_info_gain)
dt.print_tree()

# dot = visualize_tree(dt.root,'class', names)
# dot.render('decision_tree', format='png', cleanup=True)
# dot.view()


