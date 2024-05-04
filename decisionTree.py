import pandas as pd, numpy as np, math, itertools

'''
    Ideia: mudar o children para um dicionario assim no nó pai temos no dicionario condição: subtree
    Assim escusamos de usar o atributo condition e parece ser mais fácil na parte do predict

    PERGUNTAR SE É PREFERIVEL TER PREVISAO MESMO QUE MAIS ERRADA OU NAO TER
'''

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')

class PreprocessData:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.copy()
        self.codification = {}
        self.continuous_features = []
        self.categoric_features = []            

    #Cat to numeric
    def to_numeric(self, feature: str) -> None:
        feature_dic = {value: idx for idx, value in enumerate(set(self.dataset[feature].values))} #guardar a codificação para depois poder fazer predict
        new_column = [feature_dic[row[feature]] for _, row in self.dataset.iterrows()] #nova coluna codificada
        self.dataset[feature] = new_column
        self.codification[feature] = feature_dic #guardar a codificação na classe

    #Cont to Discrete
    def get_insert_position(self, l: list, value) -> int:
        for idx, i in enumerate(l):
            if value <= i:
                return idx
        return len(l)
    
    def eq_frequency(self, n_classes: int, feature: str) -> None:
        split_values = [] #valores onde dar split
        
        #descobre quartis
        for i in range(1, n_classes):
            quantil = (i / n_classes) * 100 
            split_value = np.percentile( self.dataset[feature] , quantil )
            split_values.append(split_value)
            
        #lista com os valores das novas classes para cada sample
        new_column = [self.get_insert_position(split_values, row[feature]) for _, row in self.dataset.iterrows()]
        
        self.dataset[feature] = new_column

        feature_dic = {}
        
        for idx, split_value in enumerate(split_values):
            
            if idx == 0:
                feature_dic[ idx ] = [float('-inf'), split_value]
                
            elif idx == len(split_values) - 1:
                feature_dic[ idx ] = [split_values[idx - 1], split_value]
                feature_dic[ idx + 1 ] = [split_value, float('inf')]

            else:
                feature_dic[ idx ] = [split_values[idx - 1], split_value]
        
        self.codification[ feature ] = ['eq_frequency', feature_dic]

    def eq_interval_width(self, n_classes: int, feature: str) -> None:
        min = self.dataset[feature].min()
        max = self.dataset[feature].max()
        width = (max - min) / n_classes
        
        split_values = [min + i * width for i in range(1, n_classes)]
        
        new_column = [self.get_insert_position(split_values, row[feature]) for _, row in self.dataset.iterrows()]
        
        self.dataset[feature] = new_column
        
        feature_dic = {}
        
        for idx, split_value in enumerate(split_values):
            
            if idx == 0:
                feature_dic[ idx ] = [float('-inf'), split_value]
                
            elif idx == len(split_values) - 1:
                feature_dic[ idx ] = [split_values[idx - 1], split_value]
                feature_dic[ idx + 1 ] = [split_value, float('inf')]

            else:
                feature_dic[ idx ] = [split_values[idx - 1], split_value]
        
        self.codification[ feature ] = ['eq_interval_width', feature_dic]

    def k_means(self, K: int, feature: str) -> None: #mudar a classe target
        centroids = self.dataset.sample(n = K) #escolhe k pontos random do dataset
        
        mask = self.dataset['ID'].isin(centroids.ID.tolist()) 
        X = self.dataset
        
        diff = 1 #variavel para verificar quando os centroids param de  ser mudados
        j = 0 #variavel controlar se é a primeira vez no ciclo
        
        XD = X
        while(diff!=0):
            i = 1 
        
            for _, row_centroid in centroids.iterrows(): #itera cada centroid
                
                ED=[] #lista para a distancia entre cada ponto e o centroid em causa
            
                for _, row_sample in XD.iterrows(): #itera cada sample \ centroids
                    d1=( row_centroid[feature] - row_sample[feature] ) ** 2
                    d2=( row_centroid[self.target] - row_sample[self.target] ) ** 2
                    
                    d=np.sqrt(d1+d2) #distancia centroid e sample
                    ED.append(d)
                    
                X.loc[:, i] = ED #associa-se a cada sample a distancia entre o centroid e ela
                i += 1

            C=[]
            for _, row in X.iterrows(): #itera cada sample
    
                min_dist = (None, float('inf'))
                
                for i in range(1, K + 1): #passa se por K centroids
                    
                    if row[i] < min_dist[1]: #verificar qual a dist minima entre sample e centroid
                        min_dist = (i, row[i])
                        
                C.append(min_dist[0]) #guarda se o centroid mais perto daquela sample
                
            X.loc[:, feature + '_discrete'] = C #clustering
            
            centroids_new = X.groupby([feature + '_discrete']).mean()[[self.target,feature]] #calcular novos centroids fazendo a média da dist de sample e centroid
            
            if j == 0:
                diff = 1
                j += 1
                
            else:
                diff = (centroids_new[self.target] - centroids[self.target]).sum() + (centroids_new[feature] - centroids[feature]).sum() #se os novos centroides forem igual aos antigos, diff==0 e acaba
                
            centroids = X.groupby([feature + '_discrete']).mean()[[self.target,feature]] #update centroids
            centroids[feature + '_discrete'] = centroids.index
            
        centroids = centroids.drop(self.target, axis=1)
        X = X.drop(feature, axis= 1)
        X = X.drop([i for i in range(1, K+1)], axis= 1)

        feature_dic = {}
        
        for cluster, row in centroids.iterrows():
            feature_dic[ cluster ] = row[feature]

        self.codification[feature] = ['k_means', feature_dic]
        self.dataset = X
    
    #prepara o dataset para a arvore
    def prepare_dataset(self, n_classes: int= 3, func= None) -> None:
        self.target = self.dataset.columns[-1]
        if 'ID' not in self.dataset.columns:
            self.dataset['ID'] = self.dataset.index + 1
        self.to_numeric(self.target)
        for feature in self.dataset.columns[:-1]:
            if (self.dataset[feature].dtype == float or self.dataset[feature].dtype == 'int64') and feature != 'ID':
                self.continuous_features.append(feature)
                func(n_classes, feature)

            elif self.dataset[feature].dtype == object or self.dataset[feature].dtype == bool:
                self.categoric_features.append(feature)
                self.to_numeric(feature= feature)

    #prepara uma linha para ser usada na previsão
    def prepare_row(self, row: pd.Series):
        new_row = []
        for feature, value in row.items():
            
            if feature in self.continuous_features:
                type_cod, dic = self.codification[feature]

                if type_cod == 'k_means': #se o tipo de codificaçao usada for k_means

                    min_dist = [-1, float('inf')]
                    
                    for cls, centroid in dic.items():
                        dist = abs(value - centroid) #distancia do ponto ao centroid
                        if dist < min_dist[1]:
                            min_dist = [cls, dist]

                    new_row.append( min_dist[0] )
                    
                else:

                    for cls, interval in dic.items():
                        if value > interval[0] and value <= interval[1]:
                            new_row.append( cls )
                            break
            
            elif feature in self.categoric_features:
                dic = self.codification[feature]
                new_row.append( dic.get(value) )
        return new_row

dataset = PreprocessData(weather_df)
dataset.prepare_dataset(n_classes= 3, func= dataset.k_means)
print(dataset.dataset)


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

# print(precision(restaurant_df))

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

# train, test = split_representative(restaurant_df, 0.2)
# x_train, y_train = train[train.columns[:-1]], train[train.columns[-1]]
# x_test, y_test = test[test.columns[:-1]], test[test.columns[-1]]

# print(type(x_test))

# classifier = DecisionTreeClassifier()
# classifier.fit(x_train, y_train)
# classifier.print_tree()