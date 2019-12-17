

# Chemical Effect Prediction Using Chemical Fingerprints.
This repository aims at predicting chemical effects using the T.E.S.T dataset ([U.S. EPA][https://www.epa.gov/chemical-research/toxicity-estimation-software-tool-test]).

## Problem
Each dataset contains training and testing data. Where each sample is a CAS number and a chemical concentration. 

### Methods
The problem is solved in a 3 step process:
1. Using PubChem to download the raw 881 bit fingerprints for each CAS number. 
2. Training models with fingerprints as input an concetration as labels. We use two models:
⋅⋅1. Baseline: Fingerprint similarity. Estimated concentration is C_k = 1/N SUM_n C_n * S_nk, where N is the number of similar compounds to consider, C_n is the concentration for a training sample and S_nk is the similarity between compound C_n and C_k. 
⋅⋅2. Ensemble model: We use [auto-sklearn][https://automl.github.io/auto-sklearn/master/], which create an ensemble of models based on crossvalidation. These ensembles can use Decision trees, SVMs, Neural Nets. etc. For each dataset we allow for 1 hour of ensemble optimization and where maximum 10 minutes is used for an individal model.  
3. Use the learned models to predict the concetrations in the test data. We report coefficient of determination (R2), mean squared error (MSE), and mean absolute error (MAE). 

## Results
The table below show the results for each dataset.

|Dataset|Simiarity R2|Simiarity MSE|Simiarity MAE|Ensemble R2|Ensemble MSE|Ensemble MAE|
|---------------|----------------|---------------|---------------|---------------|---------------|---------------|
|LC50|0.36|1.378|0.895|0.36|1.378|0.895|
