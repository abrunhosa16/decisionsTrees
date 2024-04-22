import pandas as pd, numpy as np, matplotlib as plt
from restaurant import df

class TreeNode:
    def __init__(self, condition=None, value=None, children=[]) -> None:
        self.value = value
        self.children = children
        self.condition = condition
        
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

def max_gain(df:pd.DataFrame) -> str: #return no atributo com ganho de informçao maximo
    attributes = df.columns.to_list()
    attributes.remove('Class')
    max = (None, float('-inf'))
    for attribute in attributes:
        information_gain = information_gain(df, attribute)
        if information_gain > max[1]:
            max = (attribute, information_gain)
    return max[0]

def plurality_value(df:pd.DataFrame) -> int: #return no valor mais frequente da classe alvo, contando que esse é inteiro
    values = df['Class'].value_counts().to_dict()
    max_value = (None, float('-inf'))
    for key, value in values.items():
        if value > max_value[1]:
            max_value = (key, value)
    return max_value[0]

def decisionTree(df:pd.DataFrame, attributes:list, parent_df=None) -> TreeNode:
    if df.shape[0] == 0: #sem exemplos
        return plurality_value(df)
    elif max(list(df['Class'].value_counts())) == df.shape[0]: #samples todos com a mesma classificação
        return df['Class'][0]
    elif len(attributes) == 0: #sem atributos restantes
        return plurality_value(parent_df)
    else:
        attribute = max_gain(df)
        #adicionar o atributo como raiz de uma arvore
        for value in set(df[attribute].values):
            sub_df = df[df[attribute] == value]
            attributes = attributes.remove(attribute)
            subtree = decisionTree(sub_df, attributes, df)
            #adicionar um ramo com attribute = value e subtree = subtree
        return 
   