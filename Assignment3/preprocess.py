from scipy.io import arff
import pandas as pd

import numpy as np

def load(path):
    # 该函数只能读取连续型数据
    data = arff.loadarff('data\\'+path)
    df = pd.DataFrame(data[0])
    df = df.replace(b'?',np.nan)
    df.fillna(value = df.mode(), inplace = True)    # 用众数替换NaN

    X, y = df.values[:, :-1], df.values[:, -1:]

    y_unique = np.unique(y)
    yy = []
    for i, ii in enumerate(y):
        for j, jj in enumerate(y_unique):
            if ii == jj:
                yy.append(j)
    y = np.array(yy)


    return X, y


