

from keras.layers import Embedding, Conv2D, Dense, Dropout, BatchNormalization, Flatten, Input, Lambda, MaxPooling2D, Reshape
from keras.models import Sequential, Model

from keras.constraints import MaxNorm

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras import backend as K

from pubchempy import get_compounds, Compound
import pandas as pd

import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import matplotlib.pyplot as plt
import os

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, \
    SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS


def tanimoto(fp1, fp2):
    fp1 = int(fp1, 16)
    fp2 = int(fp2, 16)
    fp1_count = bin(fp1).count('1')
    fp2_count = bin(fp2).count('1')
    both_count = bin(fp1 & fp2).count('1')
    return float(both_count) / (fp1_count + fp2_count - both_count)

class SimiarityModel(AutoSklearnRegressionAlgorithm):
    """
    Find nearest chemicals and return weighted average of them.
    """
    def __init__(self, top_n, random_state=None):
        self.top_n = top_n
        
    def fit(self, X, y):
        self.data = {'0b'+''.join([str(int(a)) for a in x]):yt for x,yt in zip(X,y)}
        return self

    def predict(self, X):
        
        prediction = []
        for x in X:
            x = '0b'+''.join([str(int(a)) for a in x])
            tmp = [tanimoto(x,k) for k in self.data]
            tmp = [(x,y) for x,y in zip(self.data.keys(),tmp)]
            tmp = sorted(tmp, reverse=True, key=lambda x: x[1])
            if self.top_n:
                tmp = tmp[:min(self.top_n,len(self.data))]
            a = [self.data[k[0]] for k in tmp]
            w = [k[1] for k in tmp]
            y = np.average(a, weights=w, axis =0)
            prediction.append(y)
        
        return np.asarray(prediction)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SM',
                'name': 'Fingerprint Simiarity Model',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE,),
                'output': PREDICTIONS}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        top_n = UniformIntegerHyperparameter(
            name='top_n', lower=1, upper=50, default_value=10
        )
        cs.add_hyperparameters([top_n])
        return cs

class FDAModel(AutoSklearnRegressionAlgorithm):
    """
    Find the nearest chemicals and fit a model to those.
    """
    def __init__(self, top_n, model_class = None, random_state=None):
        self.top_n = top_n
        if model_class:
            self.model_class = model_class()
        else:
            self.model_class = automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=60,
        per_run_time_limit=10)
        
    def fit(self, X, y):
        self.data = {'0b'+''.join([str(int(a)) for a in x]):yt for x,yt in zip(X,y)}
        return self

    def predict(self, X):
        
        prediction = []
        for x in X:
            x = '0b'+''.join([str(int(a)) for a in x])
            tmp = [tanimoto(x,k) for k in self.data]
            tmp = [(x,y) for x,y in zip(self.data.keys(),tmp)]
            tmp = sorted(tmp, reverse=True, key=lambda x: x[1])
            if self.top_n:
                tmp = tmp[:min(self.top_n,len(self.data))]
        
            X,Y = [[float(a) for a in k[0][2:]] for k in tmp], [self.data[k[0]] for k in tmp]
            X,Y = np.asarray(X), np.asarray(Y)
        
            model = self.model_class
            model.fit(X,Y.reshape((-1,)))
            x = np.asarray([float(a) for a in x[2:]]).reshape((1,-1))
            prediction.append(model.predict(x))
        return np.asarray(prediction)
    
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FDA',
                'name': 'FDA Model',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE,),
                'output': PREDICTIONS}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        top_n = UniformIntegerHyperparameter(
            name='top_n', lower=1, upper=50, default_value=10
        )
        cs.add_hyperparameters([top_n])
        return cs
    

def get_fingerprint(cid):
    c = Compound.from_cid(cid)
    return c.cactvs_fingerprint

def load_data_csv(filename):
    df = pd.read_csv(filename)
    
    X,y = [], []
    for uri, value in zip(df['cid'],df['y']):
        cid = uri.split('CID')[-1]
        fp = get_fingerprint(cid)
        if fp:
            fp = [int(f) for f in fp]
            X.append(fp)
            y.append(float(value))
    return X,y

def save_data(X,Y,filename):
    with open(filename,'w') as f:
        for x,y in zip(X,Y):
            s = ','.join([str(i) for i in x])
            s += '|' + str(y) + '\n'
            f.write(s)

def load_data(filename):
    X,Y = [], []
    with open(filename,'r') as f:
        for l in f:
            x,y = l.strip().split('|')
            x = [float(i) for i in x.split(',')]
            y = float(y)
            X.append(x)
            Y.append(y)
    return X,Y

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def train_test(tr_file, te_file):
    try:
        Xtr,ytr = load_data(tr_file[:-4]+'.txt')
        Xte,yte = load_data(te_file[:-4]+'.txt')
    except FileNotFoundError:
        Xtr,ytr = load_data_csv(tr_file)
        Xte,yte = load_data_csv(te_file)
        save_data(Xtr,ytr, tr_file[:-4]+'.txt')
        save_data(Xte,yte, te_file[:-4]+'.txt')
    
    # CV SimiarityModel
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr).reshape((-1,1))
    Xte, yte = np.asarray(Xte), np.asarray(yte).reshape((-1,1))
    kf = KFold(n_splits=5)
    k = 25
    out = set()
    for train, test in kf.split(Xtr):
        tmp = []
        for a in [k-k//2,k,k+k//2]: #log search trough space.
            model = SimiarityModel(top_n=a)
            model.fit(Xtr[train],ytr[train])
            p = model.predict(Xtr[test])
            tmp.append((a,sklearn.metrics.r2_score(ytr[test], p)))
        tmp = sorted(tmp, reverse=True, key=lambda x:x[1])
        out |= set(tmp)
        k,_ = tmp.pop(0)
    tmp = sorted(list(out), reverse=True, key=lambda x:x[1])
    k,_ = tmp.pop(0)
    
    model = SimiarityModel(top_n=k)
    model.fit(Xtr,ytr)
    p = model.predict(Xte)
    
    #model = FDAModel(top_n=k)
    #model.fit(Xtr,ytr)
    #p = model.predict(Xte)
    #print(sklearn.metrics.r2_score(yte, p))
    
    #autosklearn.pipeline.components.regression.add_regressor(SimiarityModel)
    #autosklearn.pipeline.components.regression.add_regressor(FDAModel)
    
    # Fit ensemble model.
    #automl = autosklearn.regression.AutoSklearnRegressor(
        #time_left_for_this_task=60,
        #per_run_time_limit=10,
        #ml_memory_limit=20000,
        #tmp_folder='./tmp/autosklearn_fp_tmp',
        #output_folder='./tmp/autosklearn_fp_out'
    #)
    #automl.fit(Xtr, ytr)
    #predictions = automl.predict(Xte)
    predictions = p
    
    return sklearn.metrics.r2_score(yte, p),sklearn.metrics.mean_squared_error(yte,p),sklearn.metrics.mean_absolute_error(yte,p),sklearn.metrics.r2_score(yte, predictions),sklearn.metrics.mean_squared_error(yte,predictions),sklearn.metrics.mean_absolute_error(yte,predictions)
    
def main():
    results = []
    d = 'data/'
    files = os.listdir(d)
    files = set([f.split('_').pop(0) for f in files])
    files = ['LC50']
    for f in tqdm(files):
        tr_file = d + f+'_train.csv'
        te_file = d+ f+'_test.csv'
        results.append((f,*train_test(tr_file, te_file)))
        
    divider = '|---------------|----------------|---------------|---------------|---------------|---------------|---------------|\n' 
    with open('results.md','w') as f:
        s = '|Dataset|Simiarity R2|Simiarity MSE|Simiarity MAE|Ensemble R2|Ensemble MSE|Ensemble MAE|' + '\n'
        f.write(s)
        f.write(divider)
        for dataset,r21,mse1,mae1,r22,mse2,mae2 in results:
            s = '|'+str(dataset)+'|'+str(round(r21,3))+'|'+str(round(mse1,3))+'|'+str(round(mae1,3))+'|'
            s += str(round(r22,3))+'|'+str(round(mse2,3))+'|'+str(round(mae2,3))+'|\n'
            f.write(s)
            f.write(divider)
        
    data = {c:r for c,r in zip(['Dataset','Simiarity R2','Simiarity MSE','Simiarity MAE','Ensemble R2','Ensemble MSE','Ensemble MAE'],zip(*results))}
    df = pd.DataFrame(data)
    df.to_csv('results.csv')

if __name__ == '__main__':
    main()
    
    
