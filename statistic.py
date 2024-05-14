import pandas as pd
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')

class Statistics:
    def __init__(self, tree: DecisionTreeClassifier, process: PreprocessData):
        self.tree = tree
        self.process = process
        self.target = self.tree.target
     

    #funçoes para PRECISION, ACCURACY, RECALL
    def evaluate_binary(self, y_test, y_pred) -> None:
        positive_label = 1
        negative_label = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        recall = 0
        precision = 0
        accuracy = 0

        for idx, pred in enumerate(y_pred):
            if pred == positive_label:
                if y_test[ idx ] == positive_label:
                    TP += 1
                if y_test[ idx ] == negative_label:
                    FP += 1
            elif pred == negative_label:
                if y_test[ idx ] == negative_label:
                    TN += 1
                elif y_test[ idx ] == positive_label:
                    FN += 1
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        accuracy = (TP + TN) /(TP + TN + FP + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = (2 * precision * recall)/(precision + recall) if precision + recall > 0 else 0
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Accuracy: ' + str(accuracy))
        print('F1 Score: ' + str(f1_score))
        print('Confusion Matrix: \n ' + str(TP) + '|' + str(FN) + '\n -+-\n ' + str(FP) + '|' + str(TN))
    
    def evaluate_non_binary(self, y_test: list, y_pred: list) -> None: #vi no site https://www.evidentlyai.com/classification-metrics/multi-class-metrics
        pred_values = set(y_pred)
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        recall = 0
        precision = 0
        accuracy = 0
        
        for value in pred_values:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for idx, pred in enumerate(y_pred):
                if pred == value:
                    if y_test[ idx ] == value: TP += 1
                    else: FP += 1
                else:
                    if y_test[ idx ] == value: FN += 1
            recall += TP / (TP + FN) if TP + FN > 0 else 0
            precision += TP / (TP + FP) if TP + FP > 0 else 0
            accuracy += (TP + TN) / (TP + TN + FP + FN)
        print('Precision: ' + str(precision / len(pred_values)))
        print('Recall: ' + str(recall / len(pred_values)))
        print('Accuracy: ' + str(accuracy / len(pred_values)))
    
    def evaluate(self, test_dataset: pd.DataFrame) -> None:
        x_test = test_dataset[ test_dataset.columns[:-1] ]
        y_test = test_dataset[ self.target ].to_list()
        y_pred = self.tree.predict(x_test)
        
        print('Test values: ' + str(y_test))
        print('Pred values: ' + str(y_pred))
        if len( set(self.process.dataset[self.target].values) ) == 2:
            self.evaluate_binary(y_test= y_test, y_pred= y_pred)
        else:
            self.evaluate_non_binary(y_test= y_test, y_pred= y_pred)