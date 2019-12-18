
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics

import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, FeatureAgglomeration
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lars, Lasso, ARDRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

from models import SimiarityModel, FDAModel
from utils import get_fingerprint, load_data_csv, load_data, save_data, write_results

def train_test(tr_file, te_file, scale_data=True, load_txt = True, simiarity_config = {}, fda_config = {}, auto_config = {}):
    if load_txt:
        Xtr,ytr = load_data(tr_file[:-4]+'.txt')
        Xte,yte = load_data(te_file[:-4]+'.txt')
    else:
        Xtr,ytr = load_data_csv(tr_file)
        Xte,yte = load_data_csv(te_file)
        save_data(Xtr,ytr, tr_file[:-4]+'.txt')
        save_data(Xte,yte, te_file[:-4]+'.txt')
    
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr).reshape((-1,1))
    Xte, yte = np.asarray(Xte), np.asarray(yte).reshape((-1,1))
    if scale_data:
        scaler = MinMaxScaler((0,1))
        ytr = scaler.fit_transform(ytr)
        yte = scaler.transform(yte)
    
    #Fit simiarity model
    if not simiarity_config:
        folds = len(Xtr)//100
        kf = KFold(n_splits=folds)
        max_k = len(Xtr)//10
        tmp1 = []
        for k in range(1,max_k):
            tmp2 = []
            for train, test in kf.split(Xtr):
                model = SimiarityModel(top_k=k)
                model.fit(Xtr[train],ytr[train])
                p = model.predict(Xtr[test])[0]
                tmp2.append(sklearn.metrics.r2_score(ytr[test], p))
            tmp1.append(tmp2)
            k += 1
        
        tmp1 = np.asarray(tmp1)
        tmp1 = np.mean(tmp1, axis = 1)
        k = np.argmax(tmp1) + 1
        simiarity_config['top_k'] = k
    
    model = SimiarityModel(**simiarity_config)
    model.fit(Xtr,ytr)
    p1 = model.predict(Xte)[0]
    
    #Fit FDA model
    if not fda_config:
        r2 = -1
        best_config = {}
        for m in [LinearRegression, LogisticRegression, RandomForestRegressor]:
            for c in [KMeans, SpectralClustering, AgglomerativeClustering, FeatureAgglomeration]:
                for k in range(2,20,2):
                    r_tmp = []
                    for train, test in kf.split(Xtr):
                        model = FDAModel(c,k,m)
                        model.fit(Xtr,ytr)
                        p = model.predict(Xte)[0]
                        t_tmp.append(sklearn.metrics.r2_score(yte,p))
                    tmp = sum(t_tmp)/len(t_tmp)
                    if tmp > r2:
                        fda_config['clustering_method'],fda_config['n_clusters'],fda_config['model_type'] = c,k,m
                        r2 = tmp
    
    model = FDAModel(**fda_config)
    model.fit(Xtr,ytr)
    p2 = model.predict(Xte)[0]
    
    #Fit ensemble model.
    automl = autosklearn.regression.AutoSklearnRegressor(**auto_config)
    automl.fit(Xtr, ytr)
    p3 = automl.predict(Xte)
    
    return sklearn.metrics.r2_score(yte, p1),sklearn.metrics.r2_score(yte, p2),sklearn.metrics.r2_score(yte, p3)
    
def main():
    results = []
    d = 'data/'
    files = os.listdir(d)
    files = set([f.split('_').pop(0) for f in files])
    for f in tqdm(files):
        tr_file = d + f+'_train.csv'
        te_file = d+ f+'_test.csv'
        try:
            res = train_test(tr_file, 
                              te_file, 
                              #simiarity_config={'top_k':10},
                              #fda_config={'n_clusters':10},
                              #auto_config={'time_left_for_this_task':120,'per_run_time_limit':30}
                              )
            results.append((f,*res))
        except FileNotFoundError:
            pass
    
    write_results('results.md', results)

if __name__ == '__main__':
    main()
    
    
