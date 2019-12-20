
from pubchempy import get_compounds, Compound
import pandas as pd
import numpy as np

def tanimoto(fp1, fp2):
    fp1 = int(fp1, 16)
    fp2 = int(fp2, 16)
    fp1_count = bin(fp1).count('1')
    fp2_count = bin(fp2).count('1')
    both_count = bin(fp1 & fp2).count('1')
    return float(both_count) / (fp1_count + fp2_count - both_count)

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


def write_results(filename, results, cols):
    i = len(results)
    divider = '|---------------' * len(cols) +'|\n' 
    with open('results.md','w') as f:
        s = '|'+'|'.join(cols) + '|\n'
        f.write(s)
        f.write(divider)
        for dataset,r21,r22,r23 in results:
            s = '|'+str(dataset)+'|'+str(round(r21,3))+'|'+str(round(r22,3))+'|'+str(round(r23,3))+'|'+'\n'
            f.write(s)
            i -= 1
            if i > 0:
                f.write(divider)

def to_tanimoto(X1,X2):
    tmp = []
    for x1 in X1:
        for x2 in X2:
            a = '0b'+''.join([str(int(x)) for x in x1])
            b = '0b'+''.join([str(int(x)) for x in x2])
            tmp.append(tanimoto(a,b))
    X = np.asarray(tmp).reshape((len(X1),len(X2)))
    return X
