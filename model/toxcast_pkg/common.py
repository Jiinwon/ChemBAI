import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

from itertools import product
from collections.abc import Iterable

from sklearn.model_selection import (
    StratifiedKFold, 
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)
from sklearn.cross_decomposition import PLSRegression


def data_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(X, y):
        train_x = X.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = X.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, train_y, test_x, test_y


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid


def CV(x, y, model, params, seed):
    skf = StratifiedKFold(n_splits = 5)
    
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    train_precision_ = []
    train_recall_ = []
    train_f1_ = []
    train_accuracy_ = []
    
    val_precision_ = []
    val_recall_ = []
    val_f1_ = []
    val_accuracy_ = []
    
    for train_idx, val_idx in skf.split(x, y):
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
        
        try:
            clf = model(random_state = seed, **params)
        except:
            clf = model(**params)
        
        # clf.fit(train_x, train_y)
        
        if model == sklearn.cross_decomposition._pls.PLSRegression:
            onehot_train_y = pd.get_dummies(train_y)
            
            clf.fit(train_x, onehot_train_y)
            
            train_pred = np.argmax(clf.predict(train_x), axis = 1)
            val_pred = np.argmax(clf.predict(val_x), axis = 1)
            
        else:
            clf.fit(train_x, train_y)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
        
        train_precision_.append(precision_score(train_y, train_pred, average = 'binary'))
        train_recall_.append(recall_score(train_y, train_pred, average = 'binary'))
        train_f1_.append(f1_score(train_y, train_pred, average = 'binary'))
        train_accuracy_.append(accuracy_score(train_y, train_pred))

        val_precision_.append(precision_score(val_y, val_pred, average = 'binary'))
        val_recall_.append(recall_score(val_y, val_pred, average = 'binary'))
        val_f1_.append(f1_score(val_y, val_pred, average = 'binary'))
        val_accuracy_.append(accuracy_score(val_y, val_pred))
        
    result = dict(zip(['params'] + train_metrics + val_metrics, 
                      [params] + [np.mean(train_precision_), 
                                  np.mean(train_recall_), 
                                  np.mean(train_f1_), 
                                  np.mean(train_accuracy_), 
                                  np.mean(val_precision_), 
                                  np.mean(val_recall_), 
                                  np.mean(val_f1_), 
                                  np.mean(val_accuracy_)]))
    
    return(result)
