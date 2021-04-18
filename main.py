import pandas as pd
import numpy as np
from libsvm.svmutil import *
from sklearn.model_selection import train_test_split
from time import sleep

# 1. get dataset
PATH = './iris.csv'

data = pd.read_csv(PATH)

# 2. Preprocessing
flower_name = data.species
flower_name = list(set(flower_name))
flower = {}
i = 0.0
for f in flower_name:
    flower[f] = i
    i += 1.0

X_pd, y_pd = data.iloc[:,:-1], data.iloc[:,[-1]]

X = np.asarray(X_pd)
y = np.asarray(y_pd)
y_pre = []
for i in y:
    y_pre.append(flower[i[0]])

X_pre = []
for row in X:
    label = 1
    row_temp = {}
    for cel in row:
        row_temp[label] = cel
        label+=1
    X_pre.append(row_temp)

# 3. train model
X_train, X_test, y_train, y_test = train_test_split(X_pre, y_pre, test_size=0.3, random_state=100)
prob = svm_problem(y_train, X_train)
param = svm_parameter('-t 2 -c 100')
result_model = svm_train(prob, param)

print()

# 4. test model
for i in range(len(y_test)):
    p_label, p_acc, p_val = svm_predict([y_test[i]], [X_test[i]], result_model)
    print(p_label, p_acc, p_val)
    print()
    sleep(0.5)
