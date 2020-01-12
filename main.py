import pandas as pd
import numpy as np

from scipy.io import arff
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from array import array
import matplotlib.pyplot as plt

#for KNN
number_of_neighbors = 5
#for cross validation
number_of_splits = 10

# for .arff datasets
# data = arff.loadarff('dataset.arff')
# df = pd.DataFrame(data[0])

data = pd.read_csv("sets/v1.csv", header=None, sep=',', skiprows=[0])
array = data.values
last_col = data.values.shape[1]

X = array[:,:last_col-2]
y = array[:,last_col-1]

# Ranks
from scipy.stats import ks_2samp, wilcoxon, ttest_ind   #mannwhitneyu, friedmanchisquare
p = []
# Kolmogorow-Smirnow
for i in range(X.shape[1]): #shape - ilość atrybutów(kolumn)
    p.append(ks_2samp(X[:,i],y).pvalue)
p = np.array(p)
ranks = np.argsort(-p)
ranked_X = X[:, ranks] # kolumny atrybutów ułożone według rankingu

# Wilcoxon
p = []
for i in range(X.shape[1]): #shape - ilość atrybutów(kolumn)
    p.append(wilcoxon(X[:,i],y).pvalue)
p = np.array(p)
ranks = np.argsort(-p)
wilcoxon_X = X[:, ranks]

# t-Student
p = []
for i in range(X.shape[1]): #shape - ilość atrybutów(kolumn)
    p.append(ttest_ind(X[:,i],y).pvalue)
p = np.array(p)
ranks = np.argsort(-p)
ttest_X = X[:, ranks]

number_of_attr = ranked_X.shape[1]

# Cross validation
skf = StratifiedKFold(n_splits=number_of_splits) #podobno 10 jest najbardziej optymalną wartością
scores = np.zeros((number_of_splits, number_of_attr-1, 3)) # 3d array
for f, (train, test) in enumerate(skf.split(ranked_X, y)):
    for i in range(number_of_attr-1):
        clf_r = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        # clf_r = GaussianNB()
        # clf_r = SVC(gamma="scale")
        # clf_r = MLPClassifier()
        clf_r.fit(ranked_X[train,:i+2], y[train])

        score_r = clf_r.score(ranked_X[test,:i+2], y[test])

        clf_wilc = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        # clf_wilc = GaussianNB()
        # clf_wilc = SVC(gamma="scale")
        # clf_wilc = MLPClassifier()
        clf_wilc.fit(wilcoxon_X[train,:i+2], y[train])

        score_wilc = clf_wilc.score(wilcoxon_X[test,:i+2], y[test])

        clf_ttest = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        # clf_ttest = GaussianNB()
        # clf_ttest = SVC(gamma="scale")
        # clf_ttest = MLPClassifier()
        clf_ttest.fit(ttest_X[train,:i+2], y[train])

        score_ttest = clf_ttest.score(ttest_X[test,:i+2], y[test])

        # print(f, i+2, "%.3f vs %.3f vs %.3f" % (score, score_r, score_reg))

        scores[f, i, 0] = score_r
        scores[f, i, 1] = score_wilc
        scores[f, i, 2] = score_ttest

mean_scores = np.mean(scores, axis=0)
print(mean_scores, mean_scores.shape)

#
plt.plot(range(number_of_attr-1), mean_scores[:,0], label='Kol-Smir')
plt.plot(range(number_of_attr-1), mean_scores[:,1], label='Wilcoxon')
plt.plot(range(number_of_attr-1), mean_scores[:,2], label='T-student')

plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig("foo.png")
