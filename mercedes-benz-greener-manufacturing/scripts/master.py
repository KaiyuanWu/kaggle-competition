import controller
import numpy as np
import time
import json
import os
def get_prefix(param, working_var):
    adjustabl_vars = ['subsample','eta', 'gamma', 'max_depth', 'pca_ncomp', 'ica_ncomp', 'svd_ncomp']
    prefix = ''
    for var in adjustabl_vars:
        if var != working_var:
            prefix += '%s-%f-'%(var, param[var])
    return prefix

def parse_prefix(filename):
    adjustabl_vars = ['subsample', 'eta', 'gamma', 'max_depth', 'pca_ncomp', 'ica_ncomp', 'svd_ncomp']
    num_adjustabl_vars = len(adjustabl_vars)
    param = {}
    fs = filename.split('-')
    for i in range(num_adjustabl_vars):
        if fs[2*i] == 'subsample' or fs[2*i] == 'eta' or fs[2*i] == 'gamma':
            param[fs[2*i]] = float(fs[2*i + 1])
        else:
            param[fs[2 * i]] = int(fs[2 * i + 1])
    return param


init_param = {'subsample': 0.95,
         'eta': 0.0005,
         'gamma': 0.000001,
         'max_depth': 4,
         'pca_ncomp': 0,
         'ica_ncomp': 0,
         'svd_ncomp': 0,
         'result_filename_prefix': '',

         'eval_metric': 'rmse',
         'silent': 1,
         'nthread': 1,
         'objective': 'reg:linear',
         'base_score': 100.66931812782134}

param_set = {'subsample': np.concatenate([np.arange(0.1, 1, (1-0.1)/20), [1]]),
'eta': np.concatenate([np.exp(np.arange(np.log(0.00001), np.log(0.1), (np.log(0.1) - np.log(0.00001))/20)),[0.1]]),
'gamma': np.concatenate([[0], np.exp(np.arange(np.log(0.0000001), np.log(10), (np.log(10) - np.log(0.0000001))/20)),[10]]),
'max_depth':np.array([2,3,4,5,6,7,8]),
'pca_ncomp': np.array([0,5,10,15,20,25,30,35,40,45,50]),
'ica_ncomp': np.array([0,5,10,15,20,25,30,35,40,45,50]),
'svd_ncomp':np.array([0,5,10,15,20,25,30,35,40,45,50])}


best_param = init_param.copy()
best_r2 = 0
curtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
niters = 10

for iter in range(niters):
    for p in param_set:
        init_param = best_param.copy()
        result_filename_prefix = get_prefix(init_param, p)
        jobs = []
        for v in param_set[p]:
            param = init_param.copy()
            param[p] = v
            param['result_filename_prefix'] = '%s-%s-%f-%d'%(result_filename_prefix, p, v, iter)
            param_file = open('jobs/%s-%s-%f-%d.param'%(result_filename_prefix, p, v, iter),'w')
            param_file.write(json.dumps(param))
            param_file.close()
            jobs.append('%s-%s-%f-%d.param'%(result_filename_prefix, p, v, iter))

        controller.run(jobs)
        #search best jobs
        results = os.listdir('results/')
        for ret in results:
            fs = ret.split('_')
            if ret.find(result_filename_prefix) != -1:
                iter_id = int(fs[14])
                if iter_id != iter:
                    continue

                r2 = float(fs[-1])
                if r2 > best_r2:
                    best_r2 = r2
                    param = param_file(ret)
                    for var in param_set:
                        best_param[var] = param[var]

        curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print ("%s, Iteration #%d , adjust %s, best r2 %f, best param = "%(curtime, iter, p, best_r2), best_param)




