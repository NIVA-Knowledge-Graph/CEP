
from pubchempy import get_compounds, Compound
import pandas as pd

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


def write_results(filename, results):
    i = len(results)
    divider = '|---------------|----------------|---------------|---------------|\n' 
    with open('results.md','w') as f:
        s = '|Dataset|Simiarity R2|FDA R2|Ensemble R2|' + '\n'
        f.write(s)
        f.write(divider)
        for dataset,r21,r22,r23 in results:
            s = '|'+str(dataset)+'|'+str(round(r21,3))+'|'+str(round(r22,3))+'|'+str(round(r23,3))+'|'+'\n'
            f.write(s)
            i -= 1
            if i > 0:
                f.write(divider)
