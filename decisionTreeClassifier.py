from node import Node
import pandas as pd, numpy as np 


class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.root = None
        self.target = None
    
    #retorna features e target
    def split_features_target(self, dataset: pd.DataFrame) -> list:
        return dataset.columns.to_list()[:-1], dataset.columns[-1]
        
    #avalia se todos os valores do target são iguais
    def same_class(self, dataset: pd.DataFrame) -> bool:
        return dataset[self.target].value_counts().max() == dataset.shape[0]
        
    def build_tree(self, dataset: pd.DataFrame, remaining_features: list, parent_dataset: pd.DataFrame) -> Node:
        y = dataset[self.target]
        
        #sem samples restantes
        if dataset.shape[0] == 0:
            return Node(leaf_value= self.calculate_leaf_value( parent_dataset[ self.target ] ))
        
        #todas as samples são da mesma classe
        elif self.same_class(dataset= dataset):
            return Node(leaf_value= self.calculate_leaf_value( dataset[ self.target ] ))
        
        #sem atributos restantes
        elif len(remaining_features) == 0:
            return Node(leaf_value= self.calculate_leaf_value( dataset[ self.target ] ))
            
        else:
            best_split = self.get_best_split(dataset= dataset, features= remaining_features) # dict com feature, info_gain, k: k_dataset
            remaining_features.remove( best_split['feature'] )
            
            children = []    
            for value, child_dataset in best_split['datasets'].items():
                subtree = self.build_tree(dataset= child_dataset, remaining_features= remaining_features, parent_dataset= dataset)
                subtree.set_condition(value) #o valor é a condição referente à feature do pai

                children.append(subtree)
                
            return Node(feature= best_split['feature'], info_gain= best_split['info_gain'], children= children)
         
    #Cálculo do info gain com entropia
    def entropy_df(self, dataset: pd.DataFrame) -> float:
        dataset = dataset[self.target]
        # Handle potential empty DataFrames or attributes with no unique values
        if len(dataset) == 0 or len(dataset.unique()) == 1:
            return 0  # Entropy is 0 for empty datasets or single-valued attributes
        # Vectorized implementation for efficiency using `groupby` and weighted entropy calculation
        value_counts = dataset.value_counts(normalize=True)
        entropy = -(value_counts * np.log2(value_counts)).sum()
        return entropy
    
    def entropy_class(self, dataset: pd.DataFrame, feature: str) -> dict:
        values_target = dataset[self.target].unique()
        values_feature = dataset[feature].unique()
        entropy_dic = {}
        
        for val in values_feature:
            entropy_val = 0
            for j in values_target:
                subset = dataset[(dataset[feature] == val) & (dataset[self.target] == j)]
                prob = len(subset) / len(dataset[dataset[feature] == val])
                if prob > 0:
                    entropy_val -= prob * np.log2(prob)
            entropy_dic[val] = entropy_val
        
        return entropy_dic
    
    def entropy_split(self, dataset: pd.DataFrame, feature: str) -> float:
        dic = self.entropy_class(dataset, feature)
        values = dic.keys()
        soma = 0
        tamanho = len(dataset[feature])
        for i in values:
            soma += (len(dataset[dataset[feature] == i])/tamanho) * dic[i]
        return soma
    
    def info_gain(self, dataset: pd.DataFrame, feature: str) -> float:
        return self.entropy_df(dataset) - self.entropy_split(dataset, feature)
    
    def max_info_gain(self, dataset: pd.DataFrame, features: list) -> tuple:
        info_gains = [self.info_gain(dataset= dataset, feature= feature) for feature in features]
        max_info_gain = max(info_gains)
        return (features[ info_gains.index(max_info_gain) ], max_info_gain)
    
    #Cálculo de gini
    def gini_class(self, dataset: pd.DataFrame, feature: str) -> dict: # Calcula o indice gini de cada valor em uma feature e retorna um dicionario com cada valor e seu respectivo resultado
        values_target = dataset[self.target].unique()
        values_feature = dataset[feature].unique()
        gini_dic = {}
        for val in values_feature:
            soma = 1
            for j in values_target:

                tamanho_val_feature = len(dataset[dataset[feature]== val])
                    
                a = dataset[(dataset[feature]== val) & (dataset[self.target] == j)].shape[0]
                soma -= (a/tamanho_val_feature)**2
            gini_dic[val] = soma
        return gini_dic

    def gini_split(self, dataset: pd.DataFrame, feature: str) -> float: # calcula o indice gini total da feature 
        dic = self.gini_class(dataset, feature)
        values = dic.keys()
        soma = 0
        tamanho = len(dataset[feature])
        for i in values:
            soma += (len(dataset[dataset[feature] == i])/tamanho) * dic[i]
        return soma

    def max_gini(self, dataset: pd.DataFrame, features: list) -> tuple: # (feature, max_gini)
        ginis = [self.gini_split(dataset, feature) for feature in features]
        final_gini = min(ginis)
        return (features[ ginis.index(final_gini) ], final_gini)
    
    #obtem o melhor split de dados
    def get_best_split(self, dataset: pd.DataFrame, features: list) -> dict: #dict com feature, info_gain e k: k_dataset
        feature, info_gain = self.max_info_gain(dataset, features)
        
        feature_values = self.original_dataset[feature].unique() #valores do original para haver sempre todos os ramos na arvore

        child_datasets = {}
        for value in feature_values: #separa em datasets filhos para cada valor da feature
            child_dataset = dataset[dataset[feature] == value]
            child_datasets[value] = child_dataset
            
        best_split = {}    
        best_split['feature'] = feature
        best_split['info_gain'] = info_gain
        best_split['datasets'] = child_datasets
        return best_split
    
    #obtem o valor de folha mais comum
    def calculate_leaf_value(self, y: pd.DataFrame) -> int:
        y = list(y)
        return max(y, key= y.count)
    
    #construçao da DT
    def fit(self, dataset: pd.DataFrame) -> None: 
        self.original_dataset = dataset
        features, self.target = self.split_features_target(dataset= dataset)
        self.root = self.build_tree(dataset= dataset, remaining_features= features, parent_dataset= dataset)
        
    def predict(self, X: pd.DataFrame) -> list:
        predictions = [self.make_prediction(row, self.root) for _, row in X.iterrows()]
        return predictions
    
    def make_prediction(self, x: pd.Series, tree: Node) -> int:
        if tree.is_leaf(): 
            return tree.leaf_value
        feature_value = x[tree.feature]
        for child in tree.children:
            if feature_value == child.condition:
                return self.make_prediction(x, child)
        print(x)
        print('No prediction')

    
        
    def print_tree(self, node: Node= None, indent= ""):
        if node is None:
            node = self.root
            
        if node.leaf_value is not None:
            print(indent + "Condition: " + str(node.condition) + "|   Leaf Value: ", node.leaf_value)
            return
        elif node.condition is None:
            print(indent + "Feature: " + node.feature)
        else:
            print(indent + "Condition: " + str(node.condition) + "|   Feature: " + node.feature)
        for child in node.children:
            self.print_tree(child, indent + "   ")