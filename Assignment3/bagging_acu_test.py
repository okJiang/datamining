import os
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np
import configparser

from preprocess import load


if __name__ == '__main__':
    p = ['mozilla4.arff', 'waveform-5000.arff', 'diabetes.arff', 'pc5.arff', 'pc1.arff']
    cl = ['naive bayes', 'decision tree', 'KNN', 'MLP', 'LinearSVM']
    for fpath in p:
        print(fpath)
        X, y = load(fpath)
        n_classes = np.arange(np.unique(y).size)
        for j in range(2,3):
            start_time = time.time()
            if j == 0:
                clf = BaggingClassifier(base_estimator=GaussianNB(),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif j == 1:
                clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, criterion='gini'),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif j == 2:
                clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3, p = 1),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif j == 3:
                clf = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(100),
                                        activation='relu', solver='adam', batch_size=128,
                                        alpha=1e-4, learning_rate_init=1e-3, learning_rate='adaptive',
                                        tol=1e-4, max_iter=200),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif j == 4:
                clf = BaggingClassifier(base_estimator=LinearSVC(penalty='l2', random_state=0, tol=1e-4),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            skf = StratifiedKFold(n_splits=10)
            skf_accuracy = []
            for train, test in skf.split(X, y):
                clf.fit(X[train], y[train])
                if n_classes.size < 3:
                    skf_accuracy.append(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1] if j != 4 else clf.decision_function(X[test]), average='micro'))
                else:
                    ytest_one_hot = label_binarize(y[test], n_classes)
                    skf_accuracy.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X[test]) if j != 4 else clf.decision_function(X[test]), average='micro'))
            accuracy = np.mean(skf_accuracy)
            print(cl[j], 'ACU:/%.3f'%accuracy)
