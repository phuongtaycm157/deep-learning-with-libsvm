import pandas as pd
import numpy as np
from libsvm.svm import *
from libsvm.svmutil import *
from sklearn.model_selection import train_test_split
from time import sleep

from preprocessing_data import *

# 1. get dataset
PATH = './images_dataset_vector.csv'

data = pd.read_csv(PATH)
# data = data.iloc[np.random.permutation(len(data))]

# 2. Preprocessing
# y, y_desc = stick_label(data.label)
y = np.asarray(data.label)
X = data.iloc[:,1:-1]
X = np.asarray(X)
X = precessing_many_x(X)

# 3. train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
prob = svm_problem(y_train, X_train)
param = svm_parameter('-t 0 -c 100')
result_model = svm_train(prob, param)

svm_save_model('classify_images.model', result_model)

# 4. test model

# 4.1 Đánh giá tổng thể mô hình.
p_label, p_acc, p_val = svm_predict(y_test, X_test, result_model)
ACC, MSE, SCC = evaluations(y_test, p_label)
print(ACC, MSE, SCC)

# 4.2 Dự đoán từng phần tử .
# for x in X_test:
#     x0, max_idx = gen_svm_nodearray(x)
#     y_pre = libsvm.svm_predict(result_model, x0)
#     print(label)
#     sleep(0.5)

