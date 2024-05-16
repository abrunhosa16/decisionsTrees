import pandas as pd, numpy as np, math, itertools

class PreprocessData:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.copy()
        self.codification = {}
        self.continuous_features = []
        self.categoric_features = []
        self.target = self.dataset.columns[-1]
        self.train = None
        self.test = None

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
    
    #divide em treino e teste
    def subsets(self, target: str) -> list:
        return [np.random.choice(self.dataset[self.dataset[target] == i].index.values.tolist(), size=len(self.dataset[self.dataset[target] == i]), replace=False) for i in self.dataset[target].unique()]

    def stratify(self, perc: float) -> None: 
        sample_test = math.ceil(perc * self.dataset.shape[0]) #calcula tamanho do test_df

        values_prop = self.dataset[self.target].value_counts(normalize= True) * sample_test 
        quantidade = values_prop.map(round) #nº de valores do target presente no test_df para manter proporçao
        
        values_sub = self.subsets(self.target)
        #listas para guardar indices dos valores de teste e treino
        acc_test = [] 
        acc_train = []
        for i in range(len(quantidade)):
            quant = quantidade[i]
            acc_test.append(values_sub[i][:quant])
            acc_train.append(values_sub[i][quant : ])
            
        #listas com os indices
        list_test = list(itertools.chain.from_iterable(acc_test))
        list_train = list(itertools.chain.from_iterable(acc_train))
        acc_test = np.random.choice(list_test, size = len(list_test), replace=False)
        acc_test, exc = acc_test[:sample_test], acc_test[sample_test:]
        list_train.extend(exc)
        self.test = self.dataset.iloc[acc_test]
        self.train = self.dataset.iloc[list_train]
    
    #prepara o dataset para a arvore
    def prepare_dataset(self, n_classes: int= 3, func= None) -> None:
        if 'ID' in self.dataset.columns:
            self.dataset = self.dataset.drop('ID', axis= 1)
        # self.to_numeric(self.target)
        for feature in self.dataset.columns:
            if self.dataset[feature].dtype == float or self.dataset[feature].dtype == 'int64': #cont
                self.continuous_features.append(feature)
                func(n_classes, feature)

            elif self.dataset[feature].dtype == object or self.dataset[feature].dtype == bool: #cat
                self.categoric_features.append(feature)
                self.to_numeric(feature= feature)

    #prepara uma linha para ser usada na previsão
    def prepare_row(self, row: pd.Series):
        new_row = []
        for feature, value in row.items():
            
            if feature in self.continuous_features:
                type_cod, dic = self.codification[feature]

                for cls, interval in dic.items():
                    if value > interval[0] and value <= interval[1]:
                        new_row.append( cls )
                        break
            
            elif feature in self.categoric_features:
                dic = self.codification[feature]
                new_row.append( dic.get(value) )
        return new_row
    
    # def k_means(self, K: int, feature: str) -> None: #mudar a classe target
    #     centroids = self.dataset.sample(n = K) #escolhe k pontos random do dataset
        
    #     mask = self.dataset['ID'].isin(centroids.ID.tolist()) 
    #     X = self.dataset
        
    #     diff = 1 #variavel para verificar quando os centroids param de  ser mudados
    #     j = 0 #variavel controlar se é a primeira vez no ciclo
        
    #     XD = X
    #     while(diff!=0):
    #         i = 1 
        
    #         for _, row_centroid in centroids.iterrows(): #itera cada centroid
                
    #             ED=[] #lista para a distancia entre cada ponto e o centroid em causa
            
    #             for _, row_sample in XD.iterrows(): #itera cada sample \ centroids
    #                 d1=( row_centroid[feature] - row_sample[feature] ) ** 2
    #                 d2=( row_centroid[self.target] - row_sample[self.target] ) ** 2
                    
    #                 d=np.sqrt(d1+d2) #distancia centroid e sample
    #                 ED.append(d)
                    
    #             X.loc[:, i] = ED #associa-se a cada sample a distancia entre o centroid e ela
    #             i += 1

    #         C=[]
    #         for _, row in X.iterrows(): #itera cada sample
    
    #             min_dist = (None, float('inf'))
                
    #             for i in range(1, K + 1): #passa se por K centroids
                    
    #                 if row[i] < min_dist[1]: #verificar qual a dist minima entre sample e centroid
    #                     min_dist = (i, row[i])
                        
    #             C.append(min_dist[0]) #guarda se o centroid mais perto daquela sample
                
    #         X.loc[:, feature + '_discrete'] = C #clustering
            
    #         centroids_new = X.groupby([feature + '_discrete']).mean()[[self.target,feature]] #calcular novos centroids fazendo a média da dist de sample e centroid
            
    #         if j == 0:
    #             diff = 1
    #             j += 1
                
    #         else:
    #             diff = (centroids_new[self.target] - centroids[self.target]).sum() + (centroids_new[feature] - centroids[feature]).sum() #se os novos centroides forem igual aos antigos, diff==0 e acaba
                
    #         centroids = X.groupby([feature + '_discrete']).mean()[[self.target,feature]] #update centroids
    #         centroids[feature + '_discrete'] = centroids.index
            
    #     centroids = centroids.drop(self.target, axis=1)
    #     X = X.drop(feature, axis= 1)
    #     X = X.drop([i for i in range(1, K+1)], axis= 1)

    #     feature_dic = {}
        
    #     for cluster, row in centroids.iterrows():
    #         feature_dic[ cluster ] = row[feature]

    #     self.codification[feature] = ['k_means', feature_dic]
    #     self.dataset = X