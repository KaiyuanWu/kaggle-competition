import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import  xgboost as xgb
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
import json
from sklearn.metrics import r2_score
from sklearn.cross_validation import  KFold
from collections import *

#Evaluate the r2_score
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2score', r2_score(y_pred=preds, y_true=labels)

class OrdLabelEncoder():
    def __init__(self):
        self.map_ = {}
    def fit(self, val, yval):
        stat = Counter(val)
        y = []
        label = []
        for v in stat:
            f = yval[val == v]
            y.append(np.mean(f))
            label.append(v)

        index = np.argsort(y)
        self.map_ = dict(zip([label[i] for i in index], range(len(index))))

    def transform(self, x):
        y = []
        for i in x:
            if i not in self.map_:
                y.append(-1)
            else:
                y.append(self.map_[i])
        return y

if __name__ == "__main__":
    param = {'silent': 1, 'eval_metric': 'rmse', 'ica_ncomp': 5, 'nthread': 5, 'base_score': 100.66931812782134,
             'subsample': .95, 'result_filename_prefix': '', 'eta': 0.001, 'svd_ncomp': 5, 'objective': 'reg:linear',
             'pca_ncomp': 5, 'max_depth': 3, 'gamma': 1e-06}
    num_boost_round = 5000

    
    #Read data
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    # train_stat = train.describe()
    # test_stat = test.describe()
    #
    # drop_cols = ['X11','X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X330', 'X347']
    # print(train_stat[drop_cols])
    # print('---------------------------')
    # print(test_stat[drop_cols])
    # quit()
    # train = train.drop(drop_cols, axis=1)
    #
    # test = test.drop(drop_cols, axis=1)

    #Encode the string variables
    # object_columns = []
    # for c in train.columns:
    #     if train[c].dtype == 'object':
    #         lbl = LabelEncoder()
    #         lbl.fit(list(train[c].values) + list(test[c].values))
    #         train[c] = lbl.transform(list(train[c].values))
    #         test[c] = lbl.transform(list(test[c].values))
    #         object_columns.append(c)

    #Encode the string variables
    object_columns = []
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = OrdLabelEncoder()
            lbl.fit(list(train[c].values), list(train['y'].values))
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))
            object_columns.append(c)


    if param['pca_ncomp'] > 0:
        pca = PCA(n_components=param['pca_ncomp'])
        all_data = train.drop(['y', 'ID'], axis=1)
        all_data = all_data.append(test.drop(['ID'], axis=1))
        pca.fit_transform(all_data)
        pca_results_train = pca.transform(train.drop(['y', 'ID'], axis=1))
        pca_results_test = pca.transform(test.drop(['ID'], axis=1))
        for i in range(1, param['pca_ncomp'] + 1):
            train['pca_' + str(i)] = pca_results_train[:, i - 1]
            test['pca_' + str(i)] = pca_results_test[:, i - 1]

    if param['ica_ncomp'] > 0:
        ica = FastICA(n_components=param['ica_ncomp'], random_state=64)
        all_data = train.drop(['y', 'ID'], axis=1)
        all_data = all_data.append(test.drop(['ID'], axis=1))
        ica.fit_transform(all_data)
        ica_results_train = ica.transform(train.drop(['y', 'ID'], axis=1))
        ica_results_test = ica.transform(test.drop(['ID'], axis=1))
        for i in range(1, param['ica_ncomp'] + 1):
            train['ica_' + str(i)] = ica_results_train[:, i - 1]
            test['ica_' + str(i)] = ica_results_test[:, i - 1]

    if param['svd_ncomp'] > 0:
        all_data = train.drop(['y', 'ID'], axis=1)
        all_data = all_data.append(test.drop(['ID'], axis=1))
        svd = TruncatedSVD(n_components=param['svd_ncomp'] , n_iter=7, random_state=64)
        svd.fit_transform(all_data)
        #svd.fit_transform(train.drop(['y', 'ID'], axis=1))
        svd_results_train = svd.transform(train.drop(['y', 'ID'], axis=1))
        svd_results_test = svd.transform(test.drop(['ID'], axis=1))
        for i in range(1, param['svd_ncomp'] + 1):
            train['svd_' + str(i)] = svd_results_train[:, i - 1]
            test['svd_' + str(i)] = svd_results_test[:, i - 1]


    #print('Shape train:{}\nShape test:{}'.format(train.shape, test.shape))
    train_y = train['y']
    train_data = train.drop(['y','ID'], axis=1)
    test_data = test.drop(['ID'], axis=1)
    num_train_samples, num_train_features = train_data.shape
    num_test_samples, num_test_features = test_data.shape
    assert num_test_features==num_test_features, 'Number of features is not consistent, Train %d != Test %d '%(num_train_features, num_test_features)


    #Create cross-validation  folds

    dtrain = xgb.DMatrix(train_data, train_y)
    dtest = xgb.DMatrix(test.drop(['ID'], axis=1))
    model = xgb.train(param, dtrain=dtrain, verbose_eval = 200, num_boost_round=num_boost_round)
    print(r2_score(dtrain.get_label(), model.predict(dtrain)))

    y_pred = model.predict(dtest)
    output = pd.DataFrame({'ID': test['ID'].astype(np.int32), 'y': y_pred})
    output.to_csv("test1.csv", index=False)


