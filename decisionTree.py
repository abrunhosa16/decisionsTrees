import pandas as pd, numpy as np, math, itertools
from datasets.restaurant import restaurant_df
from datasets.weather import weather_df
from datasets.iris import iris_df

'''
    Ideia: mudar o children para um dicionario assim no nó pai temos no dicionario condição: subtree
    Assim escusamos de usar o atributo condition e parece ser mais fácil na parte do predict

    PERGUNTAR SE É PREFERIVEL TER PREVISAO MESMO QUE MAIS ERRADA OU NAO TER
'''

class Node:
    def __init__(self, feature: int= None, condition: int = None, children: list= [], info_gain: float= None, leaf_value: int= None) -> None:
        #nós de decisão
        self.feature = feature
        self.children = children
        self.info_gain = info_gain
        
        #folhas
        self.leaf_value = leaf_value
        
        #todos os nós menos a raiz
        self.condition = condition
    
    def add_child(self, child) -> None:
        self.children.append(child)
        
    def set_condition(self, condition: int) -> None:
        self.condition = condition

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __str__(self) -> str:
        return 'Feature: ' + self.feature + ' Filhos: ' + str(len(self.children))
        
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.root = None
    
    #retorna features e target
    def split_features_target(self, dataset: pd.DataFrame) -> list:
        return dataset.columns[:-1], dataset.columns[-1]
        
    #avalia se todos os valores do target são iguais
    def same_class(self, dataset: pd.DataFrame, target: str) -> bool:
        return dataset[target].value_counts().max() == dataset.shape[0]
        
    def build_tree(self, dataset: pd.DataFrame, remaining_features: list, parent_dataset: pd.DataFrame) -> Node:
        _, target = self.split_features_target(dataset)
        x = dataset[remaining_features]
        y = dataset[target]
        
        if dataset.shape[0] == 0:
            return Node(leaf_value= self.calculate_leaf_value( parent_dataset[target] ))
        
        elif self.same_class(dataset= dataset, target= target):
            return Node(leaf_value= self.calculate_leaf_value( dataset[target] ))
        
        elif len(remaining_features) == 0:
            return Node(leaf_value= self.calculate_leaf_value( dataset[target] ))
            
        best_split = self.get_best_split(dataset, remaining_features) # dict com feature, info_gain, k: k_dataset
        
        if best_split['info_gain'] < 1:

            children = []    
            remaining_features.remove( best_split['feature'] )
            for value, child_dataset in best_split['datasets'].items():

                if child_dataset.empty: # caso para aquele valor nao houver nenhuma sample
                    subtree = Node(condition= value, leaf_value= self.calculate_leaf_value(y))

                else: 
                    subtree = self.build_tree(dataset= child_dataset, remaining_features= remaining_features, parent_dataset= dataset)
                    subtree.set_condition(value) #o valor é a condição referente à feature do pai

                children.append(subtree)
                
            return Node(feature= best_split['feature'], info_gain= best_split['info_gain'], children= children)
        
        return Node(leaf_value= self.calculate_leaf_value(y))     
    
    #Cálculo de gini
    def gini_class(self, dataset: pd.DataFrame, feature: str) -> dict: # Calcula o indice gini de cada valor em uma feature e retorna um dicionario com cada valor e seu respectivo resultado
        _, target = self.split_features_target(dataset)
        values_target = dataset[target].unique()
        values_feature = dataset[feature].unique()
        gini_dic = {}
        for val in values_feature:
            soma = 1
            for j in values_target:

                tamanho_val_feature = len(dataset[dataset[feature]== val])
                    
                a = len(dataset[(dataset[feature]== val) & (dataset[target] == j)])
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
    
    #Cálculo ganho de informação
    def b(self, q: float) -> float:
        return -(q*np.log2(q) + (1-q)*np.log2(1-q)) if q not in {0, 1} else 0
     
    def remainder(self, dataset: pd.DataFrame, feature: str) -> float:
        _, target = self.split_features_target(dataset)
        class_counts = dataset[target].value_counts()
        p = class_counts.get(1, 0) 
        n = class_counts.get(0, 0) 
        
        possible_values = dataset[feature].unique()
        r = 0
        
        for value in possible_values:
            k_dataset = dataset[dataset[feature] == value]
            k_class_counts = k_dataset[target].value_counts()
            pk = k_class_counts.get(1, 0)  
            nk = k_class_counts.get(0, 0) 
            r += (pk + nk) / (p + n) * self.b(pk / (pk + nk))
            
        return r

    def information_gain(self, dataset: pd.DataFrame, feature: str) -> float:
        return 1 - self.remainder(dataset, feature)

    def max_info_gain(self, dataset: pd.DataFrame) -> tuple: # (feature, info_gain)
        features, _ = self.split_features_target(dataset)
        info_gains = [self.information_gain(dataset, feature) for feature in features]
        info_gain = max(info_gains)
        return (features[ info_gains.index(info_gain) ], info_gain)

    #obtem o melhor split de dados
    def get_best_split(self, dataset: pd.DataFrame, features: list) -> dict: #dict com feature, info_gain e k: k_dataset
        feature, gain = self.max_gini(dataset, features)
        feature_values = self.original_dataset[feature].unique() #valores do original para haver sempre todos os ramos na arvore

        child_datasets = {}
        for value in feature_values: #separa em datasets filhos para cada valor da feature
            child_dataset = dataset[dataset[feature] == value]
            child_datasets[value] = child_dataset
            
        best_split = {}    
        best_split['feature'] = feature
        best_split['info_gain'] = gain
        best_split['datasets'] = child_datasets
        return best_split
    
    #obtem o valor de folha mais comum
    def calculate_leaf_value(self, y: pd.DataFrame) -> int:
        y = list(y)
        return max(y, key= y.count)
    
    #construçao da DT
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None: 
        dataset = pd.concat((x,y), axis=1)
        self.original_dataset = dataset
        features = x.columns.to_list()
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

    def subsets(self, df, target):
        return [np.random.choice(df[df[target] == i].index.values.tolist(), size=len(df[df[target] == i]), replace=False) for i in df[target].unique()]

    def stratify(self, df:pd.DataFrame, perc): # generalização do split representative working 
        _, target = self.split_features_target(df)
        sample_test = math.ceil(perc * df.shape[0])
        values_prop = df[target].value_counts(normalize= True) * sample_test
        quantidade = values_prop.map(round)
        values_sub = self.subsets(df, target)
        acc_test = []
        acc_train = []
        for i in range(len(quantidade)):
            quant = quantidade[i]
            acc_test.append(values_sub[i][:quant])
            acc_train.append(values_sub[i][quant : ])

        list_test = list(itertools.chain.from_iterable(acc_test))
        list_train = list(itertools.chain.from_iterable(acc_train))
        acc_test = np.random.choice(list_test, size = len(list_test), replace=False)
        acc_test, exc = acc_test[:sample_test], acc_test[sample_test:]
        list_train.extend(exc)
        test = df.iloc[acc_test]
        train = df.iloc[list_train]

        return train, test
        
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



def split_representative(df: pd.DataFrame, perc: float) -> float:
        _, target = DecisionTreeClassifier().split_features_target(df)
        yes_per, _ = df[target].value_counts(normalize=True) # proporção de sims e naos
        yes_sub, no_sub = DecisionTreeClassifier().subsets(df, target)[0],  DecisionTreeClassifier().subsets(df, target)[1] # o subconjunto de sins e naos representados pelo ID
        # Embaralhar os indices de sins e naos
        total_yes, total_no = np.random.choice(yes_sub, size=len(yes_sub), replace=False), np.random.choice(no_sub, size=len(no_sub), replace=False)
        num_train = math.ceil(perc * df.shape[0])
        num_yes = math.ceil(num_train * yes_per) # calcula o valor a ser removido para teste 
        num_no = num_train - num_yes # calcula o valor a ser removido para teste 
        test = df.iloc[[*total_yes[:num_yes],*total_no[:num_no]]] # indices para teste
        train = df.iloc[[*total_yes[num_yes:], *total_no[num_no:]]] # indices para treino 

        return train, test

def precision(dataframe: pd.DataFrame) -> float:
    positivos = 0
    total = 0

    for _ in range(20):
        classifier = DecisionTreeClassifier()

        train_df, test_df = classifier.stratify(dataframe, 0.2)
        x_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        x_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1].tolist()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        for j in range(len(y_pred)):
            if y_pred[j] == y_test[j]:
                positivos += 1
            total += 1
    return positivos/total

print(precision(restaurant_df))

def entropy(a) -> float:
    # Handle potential empty DataFrames or attributes with no unique values
    if len(a) == 0 or len(a.unique()) == 1:
        return 0  # Entropy is 0 for empty datasets or single-valued attributes

    # Vectorized implementation for efficiency using `groupby` and weighted entropy calculation
    value_counts = a.value_counts(normalize=True)
    entropy = -(value_counts * np.log2(value_counts)).sum()
    return entropy



def point_split(df, attribute, extremos): # função para avaliar onde fazer a melhor divisao (para binario) em classes contínuas  not working 
    acc = []
    b = df[attribute]
    minorante, majorante = extremos
    num_min, num_max = df[attribute].min(), df[attribute].max()
    ran = num_max - num_min
    ran = int(ran/2)
    for i in range(ran):
        a = pd.cut(b, bins=[minorante, num_min + i, num_max - i, majorante + 1], labels=[0,1, 2])
        #print(a)
    #minimo = min(acc)
    m, n = acc.index(max(acc)) + num_min, num_max - acc.index(max(acc))
    print(m,n)
    print(acc)
    return 
#print(point_split(weather_df, 'Temp', (50, 100)))

train, test = split_representative(restaurant_df, 0.2)
x_train, y_train = train[train.columns[:-1]], train[train.columns[-1]]
x_test, y_test = test[test.columns[:-1]], test[test.columns[-1]]

print(type(x_test))

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
classifier.print_tree()