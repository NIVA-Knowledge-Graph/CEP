
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

from models import SimiarityModel, FDAModel, tanimoto
from utils import get_fingerprint, load_data_csv, load_data, save_data, write_results

from collections import namedtuple

from sklearn.metrics import silhouette_score
import json
import argparse
from utils import to_tanimoto

def find_n_clusters(method, max_n, X, tanimoto_based):
    if tanimoto_based:
        X = to_tanimoto(X,X)
    sil = []
    K = range(2,max_n+1)
    curr_score = 0
    for k in K:
        try:
            labels = method(n_clusters = k).fit_predict(X)
            curr_score = silhouette_score(X, labels, metric = 'euclidean')
            sil.append(curr_score)
        except:
            sil.append(0)
        
        if len(sil) > 10 and curr_score < sil[-10]:
            break
        
    m = np.argmax(np.asarray(sil))
    return K[m]

def cv(model, X, Y):
    folds = len(X)//100
    kf = KFold(n_splits=folds)
    tmp = []
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        r2 = sklearn.metrics.r2_score(Y[test], model.predict(X[test])[0])
        tmp.append(r2)
    return sum(tmp)/len(tmp)

def predictions(tr_file, scale_data = True, load_txt = True, simiarity_config = {}, fda_config = {}, auto_config = {}):
    if load_txt:
        Xtr,ytr = load_data(tr_file[:-4]+'.txt')
        Xte,_ = load_data(te_file[:-4]+'.txt')
    else:
        Xtr,ytr = load_data_csv(tr_file)
        Xte,_ = load_data_csv(te_file)
        
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr).reshape((-1,1))
    Xte, yte = np.asarray(Xte), np.asarray(yte).reshape((-1,1))
    if scale_data:
        scaler = MinMaxScaler((0,1))
        ytr = scaler.fit_transform(ytr)
    
    model = SimiarityModel(**simiarity_config)
    model.fit(Xtr,ytr)
    p1 = model.predict(Xte)[0]
    
    model = FDAModel(**fda_config)
    model.fit(Xtr,ytr)
    p2 = model.predict(Xte)[0]
    
    #Fit ensemble model.
    if 'tanimoto_based' in auto_config and auto_config['tanimoto_based']:
        Xtr = to_tanimoto(Xtr, Xtr)
        Xte = to_tanimoto(Xte, Xtr)
        auto_config.pop('tanimoto_based', None)
        
    automl = autosklearn.regression.AutoSklearnRegressor(**auto_config)
    automl.fit(Xtr, ytr)
    p3 = automl.predict(Xte)
    
    if scale_data:
        p1 = scaler.inverse_transform(p1)
        p2 = scaler.inverse_transform(p2)
        p3 = scaler.inverse_transform(p3)
    
    return p1,p2,p3
    
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
        best_config = None,
        best_r2 = -1
        max_k = len(Xtr)//10
        tmp1 = []
        for k in range(1,max_k):
            simiarity_config['top_k'] = k
            model = SimiarityModel(**simiarity_config)
            r2 = cv(model, Xtr, ytr)
            
            if r2 > best_r2:
                best_r2 = r2
                best_config = simiarity_config
        simiarity_config = best_config
    
    model = SimiarityModel(**simiarity_config)
    model.fit(Xtr,ytr)
    p1 = sklearn.metrics.r2_score(yte, model.predict(Xte)[0])
   
    #Fit FDA model
    if not fda_config:
        # find optimal n_clusters for each clustering method
        fda_configs = []
        for tb in [True,False]:
            for c in [KMeans]:
                k = find_n_clusters(c,max_n=len(Xtr)//5,X=Xtr, tanimoto_based=tb)
                fda_configs.append({'clustering_method':c,'n_clusters':k,'tanimoto_based':tb})
        
        # find best model to go with clustering methods.
        best_r2 = -1
        best_config = {}
        for m in [LinearRegression, RandomForestRegressor]:
            for fda_config in fda_configs:
                fda_config['model_type'] = m
                model = FDAModel(**fda_config)
                r2 = cv(model,Xtr,ytr)
                if r2 > best_r2:
                    best_config = fda_config
                    best_r2 = r2
        fda_config = best_config
    
    model = FDAModel(**fda_config)
    model.fit(Xtr,ytr)
    p2 = sklearn.metrics.r2_score(yte, model.predict(Xte)[0])
    
    #Fit ensemble model.
    if not auto_config:
        best_r2 = -1
        best_config = {}
        best_tb = False
        for tb in [True, False]:
            if tb:
                tmpXtr = to_tanimoto(Xtr, Xtr)
                tmpXte = to_tanimoto(Xte, Xtr)
            for ft in ['Categorical', 'Numerical']:
                feat_type = [ft] * len(tmpXtr[0])
                
                automl = autosklearn.regression.AutoSklearnRegressor(**{'time_left_for_this_task':600,'per_run_time_limit':60})
                automl.fit(tmpXtr, ytr, feat_type=feat_type)
                r2 = sklearn.metrics.r2_score(yte, automl.predict(tmpXte))
                if r2 > best_r2:
                    best_feat = feat_type
                    best_tb = tb
                    
        auto_config['feat_type'] = best_feat
        auto_config['tanimoto_based'] = best_tb
        
    if 'tanimoto_based' in auto_config:
        if auto_config.pop('tanimoto_based',None):
            tmpXtr = to_tanimoto(Xtr,Xtr)
            tmpXte = to_tanimoto(Xte,Xtr)
        else:
            tmpXtr = Xtr 
            tmpXte = Xte
    else:
        tmpXtr = Xtr 
        tmpXte = Xte
    
    if 'feat_type' in auto_config:
        feat_type = auto_config.pop('feat_type', None)
    else:
        feat_type = None
        
    automl = autosklearn.regression.AutoSklearnRegressor(**auto_config)
    automl.fit(tmpXtr, ytr, feat_type=feat_type)
    p3 = sklearn.metrics.r2_score(yte, automl.predict(tmpXte))
    
    configs = {}
    configs['simiarity'] = simiarity_config
    configs['fda'] = fda_config
    configs['ensemble'] = auto_config
    
    return configs, (p1,p2,p3)

def main(cv = False, datasets = None, configs = None, prediction = False, load_txt = False, scale_data = True):
    if not cv and not configs:
        raise NotImplementedError
    
    simiarity_config = {}
    fda_config = {}
    auto_config = {'time_left_for_this_task':60,'per_run_time_limit':10}
    if configs and not cv:
        with open(configs) as json_file:
            data = json.load(json_file)
            if 'simiarity' in data:
                simiarity_config = data['simiarity']
            if 'fda' in data:
                fda_config = data['fda']
                for k in fda_config:
                    if isinstance(fda_config[k],dict):
                        c,tmp = fda_config[k].keys()
                        fda_config[k] = namedtuple(fda_config[k][c[0]], c[1:])(*fda_config[k][c[1:]])
                        print(fda_config[k])
            if 'ensemble' in data:
                auto_config = data['ensemble']
    
    results = []
    d = 'data/'
    if not datasets:
        files = os.listdir(d)
        files = set([f.split('_').pop(0) for f in files])
        datasets = files
        
    for f in tqdm(datasets):
        if prediction:
            tr_file = d + f+'_train.csv'
            te_file = d+ f+'_test.csv'
            res = predictions(tr_file, scale_data = scale_data, load_txt = load_txt, simiarity_config=simiarity_config, fda_config=fda_config, auto_config=auto_config)
            p1,p2,p3 = res
            data = {'Simiarty':p1, 'FDA':p2, 'Ensemble':p3}
            df = pd.DataFrame(data)
            df.to_csv(f+'_prediction.csv')
            
        else:
            tr_file = d + f+'_train.csv'
            te_file = d+ f+'_test.csv'
            configs,res = train_test(tr_file, 
                                te_file, 
                                load_txt = load_txt, 
                                simiarity_config=simiarity_config,
                                fda_config=fda_config,
                                auto_config=auto_config,
                                scale_data = scale_data
                                )
            results.append((f,*res))
            
            if cv:
                try:
                    with open('configs/'+f+'_config.txt', 'w') as outfile:
                        json.dump(configs, outfile, default=lambda x: x.__dict__)
                except:
                    pass
                    
    if not prediction:
        write_results('results.md', results, cols = ['Dataset','Simiarity R2','FDA R2','Ensemble R2'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chemical Effect Prediction Using Chemical Fingerprints.')
    
    parser.add_argument('-d','--datasets', nargs='+', help='Datasets (if empty: all datasets)', required=False)
    parser.add_argument('-c','--config', help='Config file. See LC50_config.txt for example', required=False)
    parser.add_argument('--cv', action='store_true', default=False,
                    help='Run Cross Validation')
    parser.add_argument('--fp', action='store_true', default=False,
                    help='Fetch fingerprints from PubChem, and save to txt for faster execution later.')
    parser.add_argument('--sd', action='store_true', default=False,
                    help='Scale labels.')
    parser.add_argument('--p',action='store_true', default=False, help='Predict mode, will output file dataset_prediction.txt. Provide test file with fingerprints/CIDs and labels (these are ignored).')

    args = parser.parse_args()
    main(cv=args.cv, datasets = args.datasets, configs = args.config, prediction=args.p, load_txt=not args.fp, scale_data=args.sd)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
