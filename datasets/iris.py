import pandas as pd

iris_df = pd.read_csv('datasets/iris.csv')

iris_df = iris_df.drop('ID', axis=1)

class_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_df['class'] = iris_df['class'].map(class_dict)


print(iris_df)