import pandas as pd
import pandas
import numpy as np
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from array import array
import matplotlib.pyplot as plt
#Metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from datasets import datasetsdict

#
# data = pd.read_csv("anemia.csv", header=None, sep=';')
#
# array = data.values
# X = array[:,1:]       # wszystkie wiersze, kolumnny od 1

# y = array[:,0]        # wszystkie wiersze, kolumna 0
# print(X)
#for KNN
number_of_neighbors = 5
#for cross validation
number_of_splits = 10

#
# set = datasetsdict.get("Dermatology")
# print(set)

data = pd.read_csv("sets/dataset_36_segment.csv", header=None, sep=';')
# print(data)
# print(data.values.shape)
array = data.values
# print(data.values.shape)
X = array[:,1:]
y = array[:,0]

# print(X.shape[1])
# print((X[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]] == '?').sum())

# print(y.shape)

# quit()
# Ranks
from scipy.stats import ks_2samp, wilcoxon, ttest_ind   #mannwhitneyu, friedmanchisquare
p = []
for i in range(X.shape[1]): #shape - ilość atrybutów(kolumn)
    # print(i)
    p.append(ks_2samp(X[:,i],y).pvalue)
p = np.array(p)
# print(p)
ranks = np.argsort(-p)
# print(ranks)

ranked_X = X[:, ranks] # kolumn atrybutów ułożone według rankingu
number_of_attr = ranked_X.shape[1]
print("Ranks: " + str(ranks))
print("Shape: " + str(number_of_attr))
# quit()
#
skf = StratifiedKFold(n_splits=number_of_splits) #podobno 10 jest najbardziej optymalną wartością
scores = np.zeros((number_of_splits, number_of_attr-1, 3)) # 3d array
for f, (train, test) in enumerate(skf.split(ranked_X, y)):
    for i in range(number_of_attr-1):
        random_scores = []
        for j in range(10):
            random_X = X[:, np.argsort(np.random.random_sample(X.shape[1]))]
            clf = KNeighborsClassifier(n_neighbors=number_of_neighbors)
            clf.fit(random_X[train,:i+2], y[train])
            random_scores.append(clf.score(random_X[test,:i+2], y[test]))
        score = np.mean(random_scores)
        # ypred = clf.predict(random_X[test,:i+2])
        # print("Test: " + str(y[test].shape) + "  Pred: " + str(ypred.shape))
        # acc_score = accuracy_score(y_true=y[test], y_pred=ypred)
        # av_prec = average_precision_score(y_true=y[test], y_score=ypred)
        # print("Mean: " + str(score) + ", ACC: " + str(acc_score) )

        clf_r = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        clf_r.fit(ranked_X[train,:i+2], y[train])

        score_r = clf_r.score(ranked_X[test,:i+2], y[test])
        # ypred_r = clf_r.predict(ranked_X[test,:i+2])
        # acc_score_r = accuracy_score(y_true=y[test], y_pred=ypred_r)

        clf_reg = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        clf_reg.fit(X[train,:i+2], y[train])
        score_reg = clf_reg.score(X[test,:i+2], y[test])
        # ypred_reg = clf_reg.predict(X[test,:i+2])
        # acc_score_reg = accuracy_score(y_true=y[test], y_pred=ypred_reg)

        print(f, i+2, "%.3f vs %.3f vs %.3f" % (score, score_r, score_reg))
        # print(f, i+2, "%.3f vs %.3f vs %.3f" % (acc_score, acc_score_r, acc_score_reg))
        scores[f, i, 0] = score
        scores[f, i, 1] = score_r
        scores[f, i, 2] = score_reg
        # scores[f, i, 0] = acc_score
        # scores[f, i, 1] = acc_score_r
        # scores[f, i, 2] = acc_score_reg

mean_scores = np.mean(scores, axis=0)
print(mean_scores, mean_scores.shape)

#
plt.plot(range(number_of_attr-1), mean_scores[:,0], label='random')
plt.plot(range(number_of_attr-1), mean_scores[:,1], label='ranked')
plt.plot(range(number_of_attr-1), mean_scores[:,2], label='regular')
# plt.plot(range(30), mean_scores[:,3], label='acc_random')
# plt.plot(range(30), mean_scores[:,4], label='acc_ranked')
# plt.plot(range(30), mean_scores[:,5], label='acc_regular')
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig("foo.png")
