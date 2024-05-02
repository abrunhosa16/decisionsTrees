import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets.restaurant import restaurant_df

class Node:
    def __init__(self, feature: str= None, condition: int= None, leaf_value: int= None, children: list= []) -> None:
        self.feature = feature 
        self.condition = condition 
        self.leaf_value = leaf_value 
        self.children = children if children is not None else []
        
    def copy(self):
        return deepcopy(self)
    
    def __str__(self) -> str:
        return str(self.feature) + ' / ' + str(self.condition) + ' / ' + str(self.leaf_value)
        
    def add_child(self, child) -> None:
        self.children.append(child)
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_children(self, children: list= []) -> None:
        self.children += children
        
    def print_tree(self, level=0) -> None:
        prefix = "|   " * level
        print(f"{prefix}|-- {self.feature} |Valor do ramo:{self.condition if self.condition is not None else '##'}|  |Classe:{self.leaf_value if self.leaf_value is not None else '##'}|")
        for child in self.children:
            child.print_tree(level + 1)
            
class DecistionTree:
    def __init__(self) -> None:
        self.root = None
        self.og_dataset = None  
    
    def entropy(self, dataset: pd.DataFrame, feature: str) -> float:
        if len(dataset) == 0 or len(dataset[feature].unique()) == 1:
            return 0 

        value_counts = dataset[feature].value_counts(normalize=True)
        entropy = -(value_counts * np.log2(value_counts)).sum()
        return entropy
    
    def conditional_entropy(self, dataset: pd.DataFrame, feature: str, target_attribute: str) -> float:
        target_classes = dataset[target_attribute].unique()
        entropy_attribute = 0
        total_examples = len(dataset)
        for cls in target_classes:
            subset_df = dataset[dataset[target_attribute] == cls]
            entropy_subset = self.entropy(subset_df, feature)
            prob_cls = len(subset_df) / total_examples 
            entropy_attribute += prob_cls * entropy_subset
        return entropy_attribute

    def information_gain(self, dataset: pd.DataFrame, feature: str) -> float: 
        return self.entropy(dataset, 'Class') - self.conditional_entropy(dataset, feature, 'Class')
    
    def max_gain(self, dataset: pd.DataFrame, features: list) -> str: 
        max_info_gain = (None, float('-inf'))
        for attribute in features:
            gain = self.information_gain(dataset, attribute)
            if gain > max_info_gain[1]:
                max_info_gain = (attribute, gain)
        return max_info_gain[0]
    
    def plurality_value(self, dataset: pd.DataFrame) -> int: 
        values = dataset['Class'].value_counts().to_dict()
        max_value = (None, float('-inf'))
        for key, value in values.items():
            if value > max_value[1]:
                max_value = (key, value)
        return max_value[0]
    
    def decisionTree(self, original_dataset: pd.DataFrame, dataset: pd.DataFrame, features: list, parent_dataset=None) -> Node:
        
        if dataset.empty: #sem exemplos
            return Node(leaf_value= self.plurality_value(parent_dataset))
        if len(dataset['Class'].unique()) == 1: #todos os exemplos têm a mesma classe
            return Node(leaf_value= dataset['Class'].iloc[0])
        if len(features) == 0: #sem atributos
            return Node(leaf_value= self.plurality_value(dataset))
        attribute = self.max_gain(dataset, features) #atributo mais importante
        tree = Node(feature= attribute) #decisionTree
        
        features.remove(attribute) #remover atributo com max info gain para a recursão
        
        possible_values = set(original_dataset[attribute].values) #todos os possiveis valores do atributo com max info gain
        
        for value in possible_values:
            sub_df = dataset[dataset[attribute] == value] #exemplos do atributo com v=value
            subtree = self.decisionTree(original_dataset= original_dataset, dataset= sub_df, features= features, parent_dataset= dataset)
            subtree.condition = value
            tree.add_child(subtree)
        return tree

attributes = restaurant_df.columns.to_list()
attributes.remove('Class')
dt = DecistionTree()
dt = dt.decisionTree(restaurant_df, restaurant_df, attributes, restaurant_df)
print(len(dt.children))