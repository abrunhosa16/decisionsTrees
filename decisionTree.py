import pandas as pd
import numpy as np
from datasets.restaurant import df

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
        string = 'Feature: ' + self.feature + ' Filhos: ' + str(len(self.children))
        return string
        
class DecisionTreeClassifier:
    def __init__(self, min_samples_split: int= 2, max_depth: int= 2) -> None:
        self.root = None
        
        #stoppping conditions
        self.min_samples_split = min_samples_split #minimo de samples para continuar a criar nós de decisão, se o numero de samples for menor cria-se um leaf_node
        self.max_depth = max_depth
        
    def build_tree(self, dataset: pd.DataFrame, cur_depth: int = 0) -> Node:
        x = dataset[dataset.columns.to_list()[:-1]] 
        y = dataset['Class']
        
        num_samples, _= x.shape
        
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth:
            best_split = self.get_best_split(dataset) #dict com feature, info_gain, datasets filhos
            
            if best_split['info_gain'] > 0:
                    children = []
                    for value, child_dataset in best_split['datasets'].items():
                        subtree = self.build_tree(dataset= child_dataset, cur_depth= cur_depth+1)
                        subtree.set_condition(value)
                        children.append(subtree)
                    return Node(feature= best_split['feature_index'], info_gain= best_split['info_gain'], children= children)
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(leaf_value= leaf_value)     
    
    def b(self, q: float) -> float:
        if q == 0 or q == 1:
            return 0
        return -(q*np.log2(q) + (1-q)*np.log2(1-q))
    
    def entropy(self, dataset: pd.DataFrame, feature: str) -> float:
        column = dataset[feature]
        # Calculate entropy for a dataset
        if column.empty or len(column.unique()) == 1:
            return 0
        value_counts = column.value_counts(normalize=True)
        entropy = -(value_counts * np.log2(value_counts)).sum()
        return entropy

    def remainder(self, dataset: pd.DataFrame, feature: str) -> float:
        r = 0
        possible_values = dataset[feature].unique()
        dic = dataset['Class'].value_counts().to_dict()
        p, n= 0, 0
        for key, item in dic.items():
                if key == 1:
                    p = item
                if key == 0:
                    n = item
        for value in possible_values:
            k_dataset = dataset[dataset[feature] == value]
            dic = k_dataset['Class'].value_counts().to_dict()
            pk, nk = 0, 0
            
            for key, item in dic.items():
                if key == 1:
                    pk = item
                if key == 0:
                    nk = item
                    
            r += (pk + nk) / (p + n) * self.b(pk / (pk + nk))
        return r

    def information_gain(self, dataset: pd.DataFrame, feature: str) -> float:
        return 1 - self.remainder(dataset, feature)

    def max_info_gain(self, dataset: pd.DataFrame) -> tuple: # (feature, info_gain)
        features = dataset.columns[:-1]
        info_gains = [self.information_gain(dataset, feature) for feature in features]
        info_gain = max(info_gains)
        return (features[info_gains.index(info_gain)], info_gain)
 
    def get_best_split(self, dataset: pd.DataFrame) -> dict: #obtem os datasets filhos
        feature, info_gain = self.max_info_gain(dataset)
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
        
    def calculate_leaf_value(self, y: pd.DataFrame) -> int:
        y = list(y)
        return max(y, key= y.count)
    
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        dataset = pd.concat((x,y), axis=1)
        self.root = self.build_tree(dataset)
        self.features = x.columns.to_list()
        
    def predict(self, X: pd.DataFrame):
        predictions = [self.make_prediction(row, self.root) for _, row in X.iterrows()]
        return predictions
    
    def make_prediction(self, x: pd.Series, tree: Node):
        if tree.leaf_value != None: 
            return tree.leaf_value
        feature_value = x[tree.feature]
        for child in tree.children:
            if feature_value == child.condition:
                return self.make_prediction(x, child)
        print('ERROR')
        
    def print_tree(self, node: Node, indent=""):
        if node is None:
            return
        if node.leaf_value is not None:
            print(indent + "Condition: " + str(node.condition) + "|   Leaf Value: ", node.leaf_value)
            return
        print(indent + "Feature:" + node.feature + "|   Condition: " + str(node.condition))
        for child in node.children:
            self.print_tree(child, indent + "   ")
        

        
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
x_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

x_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

classifier = DecisionTreeClassifier(min_samples_split= 4, max_depth= 6)
classifier.fit(x_train, y_train)
classifier.print_tree(classifier.root)

print(y_train)
print(classifier.predict(x_train))