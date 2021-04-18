import numpy as np

def stick_label(col):
  desc_list = list(set(col))
  desc = {}
  label_idx = 0.0
  for label in desc_list:
    desc[label] = label_idx
    label_idx += 1

  col_arr = np.asarray(col)
  result = []
  for cel in col:
    result.append(desc[cel])

  return (result, desc)

def precessing_one_x(X):
    # Input: [0.0, 0.0, 0.0 ...]
    label = 1
    result = {}
    for cel in x:
        result[label] = cel
        label+=1
    # Output: [{1: 0.0, 2: 0.0, 3: 0.0 ...}]
    return [result]

def precessing_many_x(X):
    # Input: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ...]
    X_pre = []
    for row in X:
        label = 1
        row_temp = {}
        for cel in row:
            row_temp[label] = cel
            label+=1
        X_pre.append(row_temp)
    # Output: [{1: 0.0, 2: 0.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0} ...]
    return X_pre