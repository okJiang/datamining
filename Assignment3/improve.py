import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from preprocess import load
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NeighborhoodComponentsAnalysis



p = ['mozilla4.arff', 'waveform-5000.arff', 'diabetes.arff', 'pc5.arff', 'pc1.arff']
path = p[0]
X, y = load(path)

n_classes = np.arange(np.unique(y).size)

# clf = DecisionTreeClassifier(random_state=0, criterion='gini')
# clf = GaussianNB()
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = MLPClassifier(hidden_layer_sizes=(100),
# 		        activation='relu',
# 		        solver='adam',
# 		        batch_size=128,
# 		        alpha=1e-4,
# 		        learning_rate_init=1e-3,
# 		        learning_rate='adaptive',
# 		        tol=1e-4,
# 		        max_iter=200)
# clf = LinearSVC(penalty='l2', random_state=0, tol=1e-4)
# clf = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=42)), ('knn', KNeighborsClassifier(n_neighbors=3))])
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)


print('cart:' + path)

skf = StratifiedKFold(n_splits=10)
skf_accuracy1 = []
skf_accuracy2 = []

for train, test in skf.split(X, y):
    nca.fit(X[train], y[train])
    knn.fit(nca.transform(X[train]), y[train])
    skf_accuracy1.append(knn.score(nca.transform(X[train]), y[train]))
    if n_classes.size < 3:
        skf_accuracy2.append(roc_auc_score(y[test], knn.predict(nca.transform(X[train])), average='micro'))
    else:
        ytest_one_hot = label_binarize(y[test], n_classes)
        skf_accuracy2.append(roc_auc_score(ytest_one_hot, knn.predict(nca.transform(X[train])), average='micro'))
        # skf_accuracy2.append(roc_auc_score(ytest_one_hot, clf.decision_function(X[test]), average='micro'))
accuracy1 = np.mean(skf_accuracy1)
accuracy2 = np.mean(skf_accuracy2)
print('accuracy:\t%.3f'  % accuracy1 + '/%.3f'% accuracy2)
# print('acu:\t%.3f'  % accuracy2)
