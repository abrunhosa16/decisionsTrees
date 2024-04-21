import pandas as pd, numpy as np, sklearn, matplotlib as plt
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

def entropy(df:pd.DataFrame, attribute:str) -> float:
    values = df[attribute].unique()
    entropy = 0

    for value in values:
        subset_df = df[df[attribute] == value]
        prob = len(subset_df) / len(df)
        entropy -= prob * np.log2(prob)
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

def information_gain(df:pd.DataFrame, attribute:str) -> float:
    return entropy(df, 'Class') + conditional_entropy(df, 'Class', attribute)
