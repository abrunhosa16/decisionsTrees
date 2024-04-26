import pandas as pd
import numpy as np
from datasets.restaurant import df
from datasets.weather import weather1

import math

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
        
    def set_condition(self, condition: int) -> None:
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
    
    def featuresTarget(self, dataset: pd.DataFrame):
        return dataset.columns[:-1], dataset.columns[-1]
        
    def build_tree(self, dataset: pd.DataFrame, cur_depth: int = 0) -> Node:
        features, target = self.featuresTarget(dataset)
        x = dataset[features] 
        y = dataset[target]
        
        num_samples, _= x.shape
        
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth:
            best_split = self.get_best_split(dataset) #dict com feature, info_gain, datasets filhos
            
            if best_split['info_gain'] > 0:
                    children = []
                    for value, child_dataset in best_split['datasets'].items():
                        subtree = self.build_tree(dataset= child_dataset, cur_depth= cur_depth+1)
                        subtree.set_condition(value)
                        children.append(subtree)
                    return Node(feature= best_split['feature'], info_gain= best_split['info_gain'], children= children)
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(leaf_value= leaf_value)     
    
    def b(self, q: float) -> float:
        if q in {0, 1}:
            return 0
        a = -(q*np.log2(q) + (1-q)*np.log2(1-q))
        return a
    
    
    def entropy(self, dataset: pd.DataFrame, feature: str) -> float:
        column = dataset[feature]
        # Calculate entropy for a dataset
        if column.empty or len(column.unique()) == 1:
            return 0
        value_counts = column.value_counts(normalize=True)
        entropy = -(value_counts * np.log2(value_counts)).sum()
        return entropy

    def remainder(self, dataset: pd.DataFrame, feature: str) -> float:
        _, target = self.featuresTarget(dataset)
        class_counts = dataset[target].value_counts()
        p = class_counts.get(1, 0)  
        n = class_counts.get(0, 0) 
        
        r = 0
        possible_values = dataset[feature].unique()
        target = self.featuresTarget(dataset)[1]
        dic = dataset[target].value_counts().to_dict()
        p, n= 0, 0
        for key, item in dic.items():
                if key == 1:
                    p = item
                if key == 0:
                    n = item
        for value in possible_values:
            k_dataset = dataset[dataset[feature] == value]
            dic = k_dataset[target].value_counts().to_dict()
            pk, nk = 0, 0
            
            for key, item in dic.items():
                if key == 1:
                    pk = item
                if key == 0:
                    nk = item
        for value in dataset[feature].unique():
            k_dataset = dataset[dataset[feature] == value]
            k_class_counts = k_dataset[target].value_counts()
            pk = k_class_counts.get(1, 0)  
            nk = k_class_counts.get(0, 0) 
            
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
        best_split['feature'] = feature
        best_split['info_gain'] = info_gain
        best_split['datasets'] = child_datasets
        return best_split
        
    def calculate_leaf_value(self, y: pd.DataFrame) -> int:
        y = list(y)
        return max(y, key= y.count)
    
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        dataset = pd.concat((x,y), axis=1)
        self.root = self.build_tree(dataset= dataset)
        self.features = x.columns.to_list() #???
        
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
        print('NO PREDICTION')
        
    def print_tree(self, node: Node, indent=""):
        if node is None:
            return
        if node.leaf_value is not None:
            print(indent + "Condition: " + str(node.condition) + "|   Leaf Value: ", node.leaf_value)
            return
        print(indent + "Feature:" + node.feature + "|   Condition: " + str(node.condition))
        for child in node.children:
            self.print_tree(child, indent + "   ")


def subsets(df, target):
        yes = df[df[target] == 1]
        no = df[df[target] == 0]
        return yes, no

def split_representative(df, target: str, perc: float) -> float:
        yes_per, _ = df[target].value_counts(normalize=True) # proporção de sims e naos
        yes_sub, no_sub = subsets(df, target)[0].index.values.tolist(),  subsets(df, target)[1].index.values.tolist()# o subconjunto de sins e naos representados pelo ID
        # Embaralhar os indices de sins e naos
        total_yes, total_no = np.random.choice(yes_sub, size=len(yes_sub), replace=False), np.random.choice(no_sub, size=len(no_sub), replace=False)
        num_train = math.ceil(perc * df.shape[0])
        num_yes = math.ceil(num_train * yes_per) # calcula o valor a ser removido para teste 
        num_no = num_train - num_yes # calcula o valor a ser removido para teste 
        test = df.iloc[[*total_yes[:num_yes],*total_no[:num_no]]] # indices para teste
        train = df.iloc[[*total_yes[num_yes:], *total_no[num_no:]]] # indices para treino 

        return train, test
        
# train_df, test_df = split_representative(df, 'Class', 0.3)
# x_train = train_df.iloc[:, :-1]
# y_train = train_df.iloc[:, -1]

# x_test = test_df.iloc[:, :-1]
# y_test = test_df.iloc[:, -1]

# classifier = DecisionTreeClassifier(min_samples_split= 7, max_depth= 30)
# classifier.fit(x_train, y_train)
# classifier.print_tree(classifier.root)

# print(y_test)


# from sklearn.metrics import precision_score

# # Suponha que você tenha os seguintes rótulos verdadeiros e previstos pelo modelo:
# y_pred = classifier.predict(x_test)
# print(y_pred)
# # Para calcular a precisão, basta chamar a função "precision_score" passando os rótulos verdadeiros e previstos como argumentos:
# precision = precision_score(y_test, y_pred)

# A precisão será um valor entre 0 e 1:
# print(precision) 

def precision(dataframe, target):
    positivos = 0
    total = 0
    for _ in range(20):
        train_df, test_df = split_representative(dataframe, target, 0.3)
        x_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        x_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1].tolist()
        classifier = DecisionTreeClassifier(min_samples_split= 8, max_depth= 30)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        for j in range(len(y_pred)):
            if y_pred[j] == y_test[j]:
                positivos += 1
            total += 1
    return positivos/total
print(precision(weather1, "Play"))
