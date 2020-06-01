from scipy.io import arff
import pandas as pd
import os
import time
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

data = arff.loadarff('data\\waveform-5000.arff')
df = pd.DataFrame(data[0])

start_time = time.time()

X, y = df.values[:, :-1], df.values[:, -1:]
# y = np.array([np.array(i[0].decode()) for i in y])

yy = []
y_unique = np.unique(y)
for i, ii in enumerate(y):
    for j, jj in enumerate(y_unique):
        if ii == jj:
            yy.append(j)
y = np.array(yy)
n_classes = np.arange(y_unique.size)

clf = DecisionTreeClassifier(random_state=0, criterion='gini')
# clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
skf = StratifiedKFold(n_splits=10)
skf_accuracy1 = []
skf_accuracy2 = []

for train, test in skf.split(X, y):
    clf.fit(X[train], y[train])
    skf_accuracy1.append(clf.score(X[test], y[test]))
    aa = clf.predict(X[test])
    if n_classes.size < 3:
        skf_accuracy2.append(roc_auc_score(y[test], clf.predict(X[test]), average='micro'))
    else:
        ytest_one_hot = label_binarize(y[test], n_classes)
        skf_accuracy2.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X[test]), average='micro'))
accuracy1 = np.mean(skf_accuracy1)
accuracy2 = np.mean(skf_accuracy2)
print(accuracy1)
print(accuracy2)
