import pandas as pd
from decisionTreeClassifier import DecisionTreeClassifier
from preProcess import PreprocessData

restaurant_df = pd.read_csv('datasets/restaurant.csv', )
weather_df = pd.read_csv('datasets/weather.csv')
iris_df = pd.read_csv('datasets/iris.csv')

class Statistics:
    def __init__(self) -> None:
        pass
    
    #funçoes para PRECISION, ACCURACY, RECALL
    def evaluate_binary(self, y_test, y_pred) -> tuple:
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
        return precision, recall, accuracy, f1_score, str(TP) + '|' + str(FN) + '\n -+-\n ' + str(FP) + '|' + str(TN)
    
    def evaluate_non_binary(self, y_test: list, y_pred: list) -> tuple:
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
        return precision / len(pred_values), recall / len(pred_values), accuracy / len(pred_values)
    
    def evaluate_once(self, tree: DecisionTreeClassifier, process: PreprocessData) -> None:
        test = process.test
        target = process.target
        x_test = test[ test.columns[:-1] ]
        y_test = test[ target ].to_list()
        y_pred = tree.predict(x_test)
        
        # print('Test values: ' + str(y_test))
        # print('Pred values: ' + str(y_pred))
        
        if len( set(process.dataset[ target ].values) ) == 2:
            precision, recall, accuracy, f1_score, confusion_matrix= self.evaluate_binary(y_test= y_test, y_pred= y_pred)
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
            print('Accuracy: ' + str(accuracy))
            print('F1 Score: ' + str(f1_score))
            print('Confusion Matrix: \n ' + confusion_matrix)
            
        else:
            precision, recall, accuracy = self.evaluate_non_binary(y_test= y_test, y_pred= y_pred)
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
            print('Accuracy: ' + str(accuracy))
    
    '''
        dataset: 1- restaurant, 2- weather, 3- iris
        f: 1- eq_frequency, 2- eq_interval_width
        gain: 1- max_info_gain, 2- max_gini
    '''
    def evaluate_n_times(self, dataset: int, n: int, n_classes: int, f: int, gain: int, perc: float= 0.2):
        precisions = []
        recalls = []
        accuracies = []
        for _ in range(n):
            datasets = [restaurant_df, weather_df, iris_df]

            process = PreprocessData(dataset= datasets[dataset - 1])
            funcs = [process.eq_frequency, process.eq_interval_width]
            process.prepare_dataset(n_classes= n_classes, func= funcs[f - 1])
            process.stratify(perc)

            dt = DecisionTreeClassifier()
            gains = [dt.max_info_gain, dt.max_gini]
            dt.fit(process= process, option= gains[ gain - 1 ])
            
            test = process.test
            target = process.target
            x_test = test[ test.columns[:-1] ]
            y_test = test[ target ].to_list()
            y_pred = dt.predict(x_test)
            
            if len( set(process.dataset[ target ].values) ) == 2:
                precision, recall, accuracy, *_ = self.evaluate_binary(y_test= y_test, y_pred= y_pred)
            else:
                precision, recall, accuracy = self.evaluate_non_binary(y_test= y_test, y_pred= y_pred)
            
            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)
            
        stats = pd.DataFrame({'precision': precisions,
                              'recall': recalls,
                              'accuracy': accuracies
                              })
        return stats.mean()