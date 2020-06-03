import os
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import configparser

from preprocess import load
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    p = ['mozilla4.arff', 'waveform-5000.arff', 'diabetes.arff', 'pc5.arff', 'pc1.arff']
    cl = ['naive bayes', 'decision tree', 'KNN', 'MLP', 'LinearSVM', 'improve of bagging knn']
    for fpath in p:

        X, Y = load(fpath)
        for i in range(5,6):
            start_time = time.time()
            if i == 0:
                clf = BaggingClassifier(base_estimator=GaussianNB(),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif i == 1:
                clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, criterion='gini'),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif i == 2:
                clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif i == 3:
                clf = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(100),
                                        activation='relu', solver='adam', batch_size=128,
                                        alpha=1e-4, learning_rate_init=1e-3, learning_rate='adaptive',
                                        tol=1e-4, max_iter=200),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif i == 4:
                clf = BaggingClassifier(base_estimator=LinearSVC(penalty='l2', random_state=0, tol=1e-4),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            elif i == 5:
                clf = BaggingClassifier(base_estimator=Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=42)), ('knn', KNeighborsClassifier(n_neighbors=3))]),
                                        n_estimators=10,
                                        max_samples=0.5,
                                        max_features=0.5)
            accuracy = cross_val_score(clf, X, Y, cv=10).mean()
            print(cl[i], ':accuracy:%.3f'%accuracy)
