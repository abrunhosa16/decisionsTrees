import pandas as pd, numpy as np, matplotlib as plt
from restaurant import df
from copy import deepcopy

class TreeNode:
    def __init__(self, attribute=None, value=None, condition=None, children = []) -> None:
        self.attribute = attribute
        self.condition = condition
        self.value = value
        self.children = children

    def copy(self):
        return deepcopy(self)

    def __str__(self) -> str:
        return str(self.attribute) + ' / ' + str(self.condition) + ' / ' + str(self.value)
        
    def add_child(self, child) -> None:
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_children(self, children=[]) -> None:
        self.children += children

def entropy(df: pd.DataFrame, attribute: str) -> float:
    if len(df) == 0 or len(df[attribute].unique()) == 1:
        return 0 

    # Implementação vetorizada para aumentar eficiencia 
    # normalize = true faz a proporção dos valores 
    value_counts = df[attribute].value_counts(normalize=True)
    entropy = -(value_counts * np.log2(value_counts)).sum()

    return entropy

def conditional_entropy(df:pd.DataFrame, attribute:str, target_attribute:str) -> float: #H( attribute | target_attribute )
    target_classes = df[target_attribute].unique()
    entropy_attribute = 0
    total_examples = len(df)

    for cls in target_classes:
        subset_df = df[df[target_attribute] == cls]
        entropy_subset = entropy(subset_df, attribute)
        prob_cls = len(subset_df) / total_examples #probabilidade desta classe acontecer P(cls)
        entropy_attribute += prob_cls * entropy_subset
    return entropy_attribute

def information_gain(df:pd.DataFrame, attribute:str, target_atribute= 'Class') -> float:
    # conditional_entropy não está invertida as variaveis?
    return entropy(df, target_atribute) + conditional_entropy(df, target_atribute, attribute)

def max_gain(df:pd.DataFrame, attributes:list) -> str: #return no atributo com ganho de informçao maximo
    max = (None, float('-inf'))
    for attribute in attributes:
        gain = information_gain(df, attribute)
        if gain > max[1]:
            max = (attribute, gain)
    return max[0]

def plurality_value(df:pd.DataFrame) -> int: #return no valor mais frequente da classe alvo, contando que esse é inteiro
    values = df['Class'].value_counts().to_dict()
    max_value = (None, float('-inf'))
    for key, value in values.items():
        if value > max_value[1]:
            max_value = (key, value)
    return max_value[0]

def decisionTree(df:pd.DataFrame, attributes:list, parent_df=None) -> TreeNode:

    #sem exemplos
    if df.shape[0] == 0:
        return TreeNode(value = plurality_value(parent_df))  # Retorna o valor mais frequente da classe do pai
    
    #todos os samples têm a mesma classificaçao
    elif max(list(df['Class'].value_counts())) == df.shape[0]:
        return TreeNode(value = df['Class'].iloc[0])  # Retorna a classe única
    
    #sem atributos restantes
    elif len(attributes) == 0: 
        return TreeNode(value = plurality_value(df))  # Retorna o valor mais frequente da classe neste nó
    
    else:
        attribute = max_gain(df, attributes) #atributo com maximo ganho de informação
        tree = TreeNode() #cria-se a arvore
        attributes.remove(attribute)
        possible_values = set(df[attribute].values) #valores que o atributo toma

        for value in possible_values:
            sub_df = df[df[attribute] == value] #df apenas com os idx em que attribute == value
            subtree = decisionTree(df = sub_df, attributes = attributes, parent_df = df) #recursão

            subtree.attribute = attribute
            subtree.condition = value

            tree.add_child(subtree)
        return tree

attributes = df.columns.to_list()
attributes.remove('Class')
decisiontree = decisionTree(df, attributes)
print(decisiontree)
print(len(decisiontree.children))