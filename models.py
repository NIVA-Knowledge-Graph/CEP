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
    def __init__(self, top_k=10, random_state=None):
        self.top_k = top_k
        
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
            if self.top_k:
                tmp = tmp[:min(self.top_k,len(self.data))]
            a = [self.data[k[0]] for k in tmp]
            w = [k[1] for k in tmp]
            y = np.average(a, weights=w, axis =0)
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
    def __init__(self, clustering_method=KMeans, n_clusters=8, model_type=LinearRegression, random_state=None):
        self.model_class = model_type()
        self.cluster_model = clustering_method(n_clusters=n_clusters)
        
    def fit(self, X, y):
        self.clustering = self.cluster_model.fit(X).labels_
        self.y = y
        return self

    def predict(self, X):
        
        prediction = []
        for p in self.cluster_model.predict(X):
            tmp = self.y[self.clustering==p]
            prediction.append(np.mean(tmp))
        
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
            name='k', lower=2, upper=20, default_value=8
        )
        cs.add_hyperparameters([clustering_method,model_type,k])
        return cs
    
