import pandas as pd, numpy as np, seaborn, matplotlib.pyplot as plt, warnings

'''
    formas de separar var. continuas: equal interval width, equal frequency, k-means
'''

warnings.filterwarnings("ignore") #para ignorar alguns warnings que nao percebi o q era

iris_df = pd.read_csv('datasets/iris.csv')

class_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_df['class'] = iris_df['class'].map(class_dict)

class PreprocessData:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.codification = {}
        self.continuous_features = []

    def get_insert_position(self, l: list, value) -> int:
        for idx, i in enumerate(l):
            if value <= i:
                return idx
        return len(l)
    
    def eq_frequency(self, n_classes: int, feature: str) -> pd.DataFrame:
        split_values = [] #valores onde dar split
        
        #descobre quartis
        for i in range(1, n_classes):
            quantil = (i / n_classes) * 100 
            split_value = np.percentile( self.dataset[feature] , quantil )
            split_values.append(split_value)
            
        #lista com os valores das novas classes para cada sample
        new_column = [self.get_insert_position(split_values, row[feature]) for _, row in self.dataset.iterrows()]
        
        new_dataset = pd.DataFrame({feature + '_discrete': new_column})
        self.dataset = pd.concat([new_dataset, self.dataset], axis=1)

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
        return self.dataset

    def eq_interval_width(self, n_classes: int, feature: str) -> pd.DataFrame:
        min = self.dataset[feature].min()
        max = self.dataset[feature].max()
        width = (max - min) / n_classes
        
        split_values = [min + i * width for i in range(1, n_classes)]
        
        new_column = [self.get_insert_position(split_values, row[feature]) for _, row in self.dataset.iterrows()]
        
        new_dataset = pd.DataFrame( {feature + '_discrete': new_column} )
        self.dataset = pd.concat( [new_dataset, self.dataset] , axis=1)
        
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
        return self.dataset

    def k_means_feature(self, K: int, feature: str): #mudar a classe target
        centroids = self.dataset.sample(n = K) #escolhe k pontos random do dataset
        
        mask = self.dataset['ID'].isin(centroids.ID.tolist()) 
        X = self.dataset[~mask] #todos as samples exceto os centroids
        
        diff = 1 #variavel para verificar quando os centroids param de  ser mudados
        j = 0 #variavel controlar se é a primeira vez no ciclo
        
        XD = X
        while(diff!=0):
            i = 1 
        
            for _, row_centroid in centroids.iterrows(): #itera cada centroid
                
                ED=[] #lista para a distancia entre cada ponto e o centroid em causa
            
                for _, row_sample in XD.iterrows(): #itera cada sample \ centroids
                    d1=( row_centroid[feature] - row_sample[feature] ) ** 2
                    d2=( row_centroid["class"] - row_sample["class"] ) ** 2
                    
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
            
            centroids_new = X.groupby([feature + '_discrete']).mean()[["class",feature]] #calcular novos centroids fazendo a média da dist de sample e centroid
            
            if j == 0:
                diff = 1
                j += 1
                
            else:
                diff = (centroids_new['class'] - centroids['class']).sum() + (centroids_new[feature] - centroids[feature]).sum() #se os novos centroides forem igual aos antigos, diff==0 e acaba
                
            centroids = X.groupby([feature + '_discrete']).mean()[["class",feature]] #update centroids
            centroids[feature + '_discrete'] = centroids.index
            
        centroids = centroids.drop('class', axis=1)
        X = X.drop(feature, axis= 1)
        X = X.drop([i for i in range(1, K+1)], axis= 1)
        return X, centroids

    def k_means(self, K: int, feature: str) -> None:
        X, centroids = self.k_means_feature(K, feature)

        feature_dic = {}
        
        for cluster, row in centroids.iterrows():
            feature_dic[ cluster ] = row[feature]

        self.codification[feature] = ['k_means', feature_dic]
        return X
        
    def discretize_dataset(self, n_classes: int, func) -> None:
        for feature in self.dataset.columns:
            if self.dataset[feature].dtype == float:
                self.continuous_features.append(feature)
                self.dataset = func(n_classes, feature)

    def discretize_row(self, row: pd.Series):
        new_row = []
        for feature in self.continuous_features: #percorre todas as features continuas

            type_cod, dic = self.codification[feature]

            if type_cod == 'k_means': #se o tipo de codificaçao usada for k_means

                min_dist = [-1, float('inf')]
                
                for cls, centroid in dic.items():
                    dist = abs(row[feature] - centroid) #distancia do ponto ao centroid
                    if dist < min_dist[1]:
                        min_dist = [cls, dist]

                new_row.append( min_dist[0] )
                
            else:

                for cls, interval in dic.items():
                    if row[feature] > interval[0] and row[feature] <= interval[1]:
                        new_row.append( cls )
                        break

        return new_row
                
process = PreprocessData(iris_df)
process.discretize_dataset(3, process.k_means)    
print(process.codification)

X = iris_df.iloc[3]
X = X.drop(['ID', 'class'])
print(X)
print(process.discretize_row(X))