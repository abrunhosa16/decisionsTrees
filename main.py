from decisionTree import run

while True:
    dataset = int(input('Qual dataset quer usar? 1- Restaurante  2- Tempo  3- Iris: '))
    if dataset in range(1,4):
        break
while True:
    n_classes = int(input('Quantas classes quer usar para separar as features contínuas? 2-5: '))
    if n_classes in range(2,6):
        break
while True:
    f = int(input('Qual função quer usar para fazer essa separação? 1- Eq. Frequency  2- Eq. Int. Width  3- K-Means: '))
    if f in range(1,4):
        break
run(dataset= dataset, n_classes= n_classes, f= f)