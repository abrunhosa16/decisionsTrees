import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from restaurant import df

class TreeNode:
    def __init__(self, attribute=None, prev_value=None, classification=None, children=None) -> None: #prev_value para o valor que o o ramo verifica quanto ao no anterior atributo
        self.attribute = attribute
        self.prev_value = prev_value
        self.classification = classification
        self.children = children if children is not None else []

    def copy(self):
        return deepcopy(self)

    def __str__(self) -> str:
        return str(self.attribute) + ' / ' + str(self.prev_value) + ' / ' + str(self.classification)
        
    def add_child(self, child) -> None:
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_children(self, children=[]) -> None:
        self.children += children
        
    def print_tree(self, level=0):
        prefix = "|   " * level
        print(f"{prefix}|-- {self.attribute}: {self.prev_value if self.prev_value is not None else ''} - {self.classification}")
        for child in self.children:
            child.print_tree(level + 1)

def entropy(df: pd.DataFrame, attribute: str) -> float:
    if len(df) == 0 or len(df[attribute].unique()) == 1:
        return 0 

    # Implementação vetorizada para aumentar eficiencia 
    # normalize = true faz a proporção dos valores 
    value_counts = df[attribute].value_counts(normalize=True)
    entropy = -(value_counts * np.log2(value_counts)).sum()

    return entropy

def conditional_entropy(df: pd.DataFrame, attribute: str, target_attribute: str) -> float:
    target_classes = df[target_attribute].unique()
    entropy_attribute = 0
    total_examples = len(df)

    for cls in target_classes:
        #subset_df = df.query(f'{target_attribute} == {cls}')
        subset_df = df[df[target_attribute] == cls]
        entropy_subset = entropy(subset_df, attribute)
        prob_cls = len(subset_df) / total_examples #probabilidade desta classe acontecer P(cls)
        entropy_attribute += prob_cls * entropy_subset
    return entropy_attribute

def information_gain(df: pd.DataFrame, attribute: str) -> float:
    return entropy(df, 'Class') - conditional_entropy(df, attribute, 'Class')

def max_gain(df: pd.DataFrame, attributes: list) -> str:
    max_info_gain = (None, float('-inf'))
    for attribute in attributes:
        gain = information_gain(df, attribute)
        if gain > max_info_gain[1]:
            max_info_gain = (attribute, gain)
    return max_info_gain[0]

def plurality_value(df: pd.DataFrame) -> int:
    values = df['Class'].value_counts().to_dict()
    max_value = (None, float('-inf'))
    for key, value in values.items():
        if value > max_value[1]:
            max_value = (key, value)
    return max_value[0]

def decisionTree(df: pd.DataFrame, attributes: list, parent_df=None) -> TreeNode:
    
    if df.empty: #sem exemplos
        return TreeNode(classification= plurality_value(parent_df))

    if len(df['Class'].unique()) == 1: #todos os exemplos têm a mesma classe
        return TreeNode(classification= df['Class'].iloc[0])

    if len(attributes) == 0: #sem atributos
        return TreeNode(classification= plurality_value(df))

    attribute = max_gain(df, attributes) #atributo mais importante
    tree = TreeNode(attribute= attribute) #decisionTree
    
    attributes.remove(attribute) #remover atributo com max info gain para a recursão
    
    possible_values = set(df[attribute].values) #todos os possiveis valores do atributo com max info gain
    
    for value in possible_values:
        sub_df = df[df[attribute] == value] #exemplos do atributo com v=value
        subtree = decisionTree(df= sub_df, attributes= attributes.copy(), parent_df= df)
        subtree.prev_value = value
        tree.add_child(subtree)
    return tree

# Example usage:
# Assuming you have a DataFrame 'df' with columns including 'Class' and other attributes
# attributes = list(df.columns)
# attributes.remove('Class')
# root_node = decisionTree(df, attributes)

# You can then traverse the tree to make predictions or visualize it as needed.


attributes = df.columns.to_list()
attributes.remove('Class')
decisiontree = decisionTree(df, attributes)
decisiontree.print_tree()

# attributes = df.columns.to_list()
# attributes.remove('Class')
# print(attributes)
# test = [1,0,0,1,2,1,0,0,1,2,0]
# cur = decisiontree.copy()
# while len(cur.children) > 0:
#     cur_attribute = cur.attribute
#     idx = attributes.index(cur_attribute)
#     for child in cur.children:
#         print(cur_attribute)
#         print(test[idx])
#         if child.prev_value == test[idx]:
#             cur = child
#             break
# print(cur)
    
'''
    Preciso pensar numa lógica para a arvore de decisão
'''