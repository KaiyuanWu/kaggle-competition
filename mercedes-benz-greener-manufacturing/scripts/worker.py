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


#Evaluate the r2_score
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2score', r2_score(y_pred=preds, y_true=labels)

if __name__ == "__main__":
    param_filename = sys.argv[2]
    param_file = open(param_filename)
    param = json.loads(param_file.readline())
    param_file.close()


    #Read data
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    drop_cols = ['X11','X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X330', 'X347']
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    #Encode the string variables
    object_columns = []

    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))
            object_columns.append(c)
    all_data = train.drop(['y', 'ID'], axis=1).append(test.drop(['ID'], axis=1))
    if param['pca_ncomp'] > 0:
        pca = PCA(n_components=param['pca_ncomp'])
        pca.fit_transform(all_data)
        pca_results_train = pca.transform(train.drop(['y', 'ID'], axis=1))
        pca_results_test = pca.transform(test.drop(['ID'], axis=1))
        for i in range(1, param['pca_ncomp'] + 1):
            train['pca_' + str(i)] = pca_results_train[:, i - 1]
            test['pca_' + str(i)] = pca_results_test[:, i - 1]

    if param['ica_ncomp'] > 0:
        ica = FastICA(n_components=param['ica_ncomp'], random_state=64)
        ica.fit_transform(all_data)
        ica_results_train = ica.transform(train.drop(['y', 'ID'], axis=1))
        ica_results_test = ica.transform(test.drop(['ID'], axis=1))
        for i in range(1, param['ica_ncomp'] + 1):
            train['ica_' + str(i)] = ica_results_train[:, i - 1]
            test['ica_' + str(i)] = ica_results_test[:, i - 1]

    if param['svd_ncomp'] > 0:
        svd =   TruncatedSVD(n_components=param['svd_ncomp'] , n_iter=7, random_state=64)
        svd.fit_transform(all_data)
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
    nfolds = 5
    kf = KFold(n=num_train_samples, n_folds=nfolds, shuffle=True)

    #Start to train model
    evals_result_array = []
    for train_index, val_index  in kf:
        dtrain = xgb.DMatrix(train_data.iloc[train_index], train_y.iloc[train_index])
        dval = xgb.DMatrix(train_data.iloc[val_index], train_y.iloc[val_index])
        evals_result = {}
        model = xgb.train(param, dtrain=dtrain, verbose_eval = 200, num_boost_round=10000, evals = [(dtrain, "Train"), (dval, "Validation")],feval=evalerror, evals_result=evals_result)
        results = {}
        results['Validation-rmse'] = evals_result['Validation']['rmse']
        results['Validation-r2score'] = evals_result['Validation']['r2score']
        results['Train-rmse'] = evals_result['Train']['rmse']
        results['Train-r2score'] = evals_result['Train']['r2score']
        evals_result_array.append(results)

    #Accumulate the training results
    results = {}

    results['Validation-rmse'] =  np.mean(np.array([r['Validation-rmse'] for r in evals_result_array]), axis=0)
    i = np.argmin(results['Validation-rmse'])
    results['Validation-rmse-min'] = [i, results['Validation-rmse'][i]]
    results['Validation-rmse']  = list(results['Validation-rmse'] )

    results['Validation-r2score'] =  np.mean(np.array([r['Validation-r2score'] for r in evals_result_array]), axis=0)
    i = np.argmax(results['Validation-r2score'])
    results['Validation-r2score-max'] = [i, results['Validation-r2score'][i]]
    results['Validation-r2score']  = list(results['Validation-r2score'] )

    results['Train-rmse'] =  list(np.mean(np.array([r['Train-rmse'] for r in evals_result_array]), axis=0))
    results['Train-r2score'] =  list(np.mean(np.array([r['Train-r2score'] for r in evals_result_array]), axis=0))

    result_filename = '%s_%f'%(param['result_filename_prefix'], results['Validation-rmse-min'][1])
    fout = open(result_filename,'w')
    fout.write(json.dumps(results))
    fout.close()
