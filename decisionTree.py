import pandas as pd
import numpy as np
from datasets.restaurant import df

'''
    MUDAR A FUNÇAO DE GANHO DE INFORMAÇÃO E VERIFICAR TODAS ESSAS
'''

class Node:
    def __init__(self, feature: int= None, condition= None, children: list= [], info_gain: float= None, leaf_value= None) -> None:
        #for decision nodes
        self.feature = feature
        self.children = children
        self.info_gain = info_gain
        
        #for leaf nodes
        self.leaf_value = leaf_value
        
        #for all nodes except root
        self.condition = condition
    
    def add_child(self, child) -> None:
        self.children.append(child)
        
    def set_condition(self, condition) -> None:
        self.condition = condition
        
    def __str__(self) -> str:
        string = self.feature + str(len(self.children))
        return string
        
class DecisionTreeClassifier:
    def __init__(self, min_samples_split= 2, max_depth= 2) -> None:
        self.root = None
        
        #stoppping conditions
        self.min_samples_split = min_samples_split #minimo de samples para continuar a criar nós de decisão, se o numero de samples for menor cria-se um leaf_node
        self.max_depth = max_depth
        
    def build_tree(self, dataset: pd.DataFrame, curr_depth= 0):
        y = dataset['Class']
        x = dataset[dataset.columns.to_list()[:-1]]
        
        num_samples, num_features = x.shape
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features) #dict com feature, info_gain, datasets filhos
            if best_split['info_gain'] > 0:
                    #logica de recursão usando os best_split 
                    parent_node = Node(feature= best_split['feature_index'], info_gain= best_split['info_gain'])
                    for value, child_dataset in best_split['datasets'].items():
                        subtree = self.build_tree(dataset= child_dataset, curr_depth= curr_depth+1)
                        subtree.set_condition(value)
                        parent_node.add_child(subtree)
                        
                    return parent_node
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(leaf_value= leaf_value)     
    
    def entropy(self, dataset):
        # Calculate entropy for a dataset
        if dataset.empty or len(dataset.unique()) == 1:
            return 0
        value_counts = dataset.value_counts(normalize=True)
        entropy = -(value_counts * np.log2(value_counts)).sum()
        return entropy

    def conditional_entropy(self, dataset, attribute, target_attribute):
        target_classes = dataset[target_attribute].unique()
        entropy_attribute = 0
        total_examples = len(dataset)
        for cls in target_classes:
            subset_df = dataset[dataset[target_attribute] == cls]
            entropy_subset = self.entropy(subset_df[attribute])
            prob_cls = len(subset_df) / total_examples
            entropy_attribute += prob_cls * entropy_subset
        return entropy_attribute

    def information_gain(self, dataset, attribute, target_attribute):
        return self.entropy(dataset[target_attribute]) - self.conditional_entropy(dataset, attribute, target_attribute)

    def max_info_gain(self, dataset, target_attribute):
        print(dataset)
        max_info_gain = (None, float('-inf'))
        for feature in dataset.columns:
            if feature != target_attribute:
                info_gain = self.information_gain(dataset, feature, target_attribute)
                if info_gain > max_info_gain[1]:
                    max_info_gain = (feature, info_gain)
        return max_info_gain
 
    def get_best_split(self, dataset: pd.DataFrame, num_samples, num_features) -> dict: #obtem os datasets filhos
        feature, info_gain = self.max_info_gain(dataset, 'Class')
        feature_values = dataset[feature].unique()
        child_datasets = {}
        for value in feature_values: #separa em datasets filhos para cada valor da feature
            child_dataset = dataset[dataset[feature] == value]
            child_datasets[value] = child_dataset
            
        best_split = {}    
        best_split['feature_index'] = feature
        best_split['info_gain'] = info_gain
        best_split['datasets'] = child_datasets
        return best_split
        
    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key= y.count)
    
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        dataset = pd.concat((x,y), axis=1)
        self.root = self.build_tree(dataset)
        self.features = x.columns.to_list()
        
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree: Node):
        if tree.leaf_value != None: 
            return tree.leaf_value
        feature_value = x[tree.feature]
        for child in tree.children:
            if feature_value == child.condition:
                return self.make_prediction(x, child)
        print('ERROR')
        

        
        
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 41)

classifier = DecisionTreeClassifier(min_samples_split= 1, max_depth= 4)
classifier.fit(x, y)
print(classifier.root)