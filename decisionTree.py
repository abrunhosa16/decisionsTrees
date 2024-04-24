import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets.restaurant import df

class TreeNode:
    def __init__(self, attribute=None, prev_value=None, classification=None, children=None) -> None:
        self.attribute = attribute #atributo a ser avaliado no nó SE NÃO FOR FOLHA
        self.prev_value = prev_value #valor do atributo do nó pai, como se fosse o valor do ramo SE NÃO FOR A RAIZ
        self.classification = classification #classificação da classe final SE FOR FOLHA
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
        
    def print_tree(self, level=0) -> None:
        prefix = "|   " * level
        print(f"{prefix}|-- {self.attribute} |Valor do ramo:{self.prev_value if self.prev_value is not None else '##'}|  |Classe:{self.classification if self.classification is not None else '##'}|")
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

def conditional_entropy(df: pd.DataFrame, attribute: str, target_attribute: str) -> float: #H(attribute | target_attribute)
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

def information_gain(df: pd.DataFrame, attribute: str) -> float: #return no ganho de informação
    return entropy(df, 'Class') - conditional_entropy(df, attribute, 'Class')

def max_gain(df: pd.DataFrame, attributes: list) -> str: #para saber qual o atributo dos attirubtes passados com o maior ganho de informação 
    max_info_gain = (None, float('-inf'))
    for attribute in attributes:
        gain = information_gain(df, attribute)
        if gain > max_info_gain[1]:
            max_info_gain = (attribute, gain)
    return max_info_gain[0]

def plurality_value(df: pd.DataFrame) -> int: #obtem a classe mais frequente no df
    values = df['Class'].value_counts().to_dict()
    max_value = (None, float('-inf'))
    for key, value in values.items():
        if value > max_value[1]:
            max_value = (key, value)
    return max_value[0]

def decisionTree(original_df: pd.DataFrame, df: pd.DataFrame, attributes: list, parent_df=None) -> TreeNode:
    
    if df.empty: #sem exemplos
        return TreeNode(classification= plurality_value(parent_df))

    if len(df['Class'].unique()) == 1: #todos os exemplos têm a mesma classe
        return TreeNode(classification= df['Class'].iloc[0])

    if len(attributes) == 0: #sem atributos
        return TreeNode(classification= plurality_value(df))

    attribute = max_gain(df, attributes) #atributo mais importante
    tree = TreeNode(attribute= attribute) #decisionTree
    
    attributes.remove(attribute) #remover atributo com max info gain para a recursão
    
    possible_values = set(original_df[attribute].values) #todos os possiveis valores do atributo com max info gain
    
    for value in possible_values:
        sub_df = df[df[attribute] == value] #exemplos do atributo com v=value
        subtree = decisionTree(original_df= original_df, df= sub_df, attributes= attributes.copy(), parent_df= df)
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
decisiontree = decisionTree(original_df=df , df=df, attributes=attributes)
decisiontree.print_tree()

def test_restaurant_decision_tree(df: pd.DataFrame, tree: TreeNode, sample: list): #sample = [1,0,0,1,1,2,0,1,0,0,1]
    attributes = df.columns.to_list()
    attributes.remove('Class')
    cur = tree.copy()
    while len(cur.children) > 0:
        idx = attributes.index(cur.attribute)
        for child in cur.children:
            if child.prev_value == sample[idx]:
                cur = child
                break  
    return cur

print(test_restaurant_decision_tree(df, decisiontree, [1,0,0,1,1,2,0,1,0,0,1]))