### models.py

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, FeatureAgglomeration
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lars, Lasso, ARDRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, \
    SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

from utils import to_tanimoto, tanimoto


class SimiarityModel(AutoSklearnRegressionAlgorithm):
    """
    Find nearest chemicals and return weighted average of them.
    """
    def __init__(self, top_k=10, random_state=None):
        self.top_k = top_k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        
        tmp = to_tanimoto(X,self.X)
        
        prediction = []
        
        for x in X:
            tmp = to_tanimoto([x],self.X)[0]
            idx = sorted(range(len(tmp)),reverse=True,key=lambda i:tmp[i])
            if self.top_k:
                idx = idx[:self.top_k]
            
            a = self.y[idx]
            w = tmp[idx]
            y = np.average(a, weights=w, axis=0)
            prediction.append(y)
        
        return (np.asarray(prediction),)

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
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        top_n = UniformIntegerHyperparameter(
            name='top_k', lower=1, upper=50, default_value=10
        )
        cs.add_hyperparameters([top_n])
        return cs

class FDAModel(AutoSklearnRegressionAlgorithm):
    """
    Find the nearest chemicals and fit a model to those.
    """
    def __init__(self, clustering_method=KMeans, n_clusters=8, model_type=LinearRegression, tanimoto_based=False, random_state=None):
        self.model_class = model_type
        self.cluster_model = clustering_method(n_clusters=n_clusters)
        self.tanimoto_based = tanimoto_based
        
    def fit(self, X, y):
        self.Xtr = X
        self.X = X
        if self.tanimoto_based:
            self.X = to_tanimoto(self.X,self.X)
        self.clustering = self.cluster_model.fit_predict(self.X)
        self.y = y
        return self

    def predict(self, X):
        if self.tanimoto_based:
            X = to_tanimoto(X,self.Xtr)
            
        prediction = []
        for p,x in zip(self.cluster_model.predict(X),X):
            model = self.model_class()
            tmp = self.clustering==p
            model.fit(self.X[tmp],self.y[tmp])
            p = model.predict(x.reshape((1,-1)))[0]
            prediction.append(p)
        
        return (np.asarray(prediction),)
    
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
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        clustering_method = CategoricalHyperparameter(
            name='clustering_method',
            choices=[KMeans, 
                     SpectralClustering, 
                     AgglomerativeClustering, 
                     FeatureAgglomeration
                     ],
            default_value=KMeans
        )
        model_type = CategoricalHyperparameter(
            name='model_type',
            choices=[RandomForestRegressor,
                     LinearRegression,
                     LogisticRegression,
                     ElasticNet, 
                     Lars, 
                     Lasso, 
                     ARDRegression, 
                     BayesianRidge
                     ],
            default_value=RandomForestRegressor
        )
        k = UniformIntegerHyperparameter(
            name='n_clusters', lower=2, upper=20, default_value=8
        )
        cs.add_hyperparameters([clustering_method,model_type,k])
        return cs
    
