'''
###############
### DEBoost ###
###############
Author: Wei Hao Khoong
Email: khoongweihao@u.nus.edu
LinkedIn: https://www.linkedin.com/in/wei-hao-khoong-6b94b1101
Kaggle: https://www.kaggle.com/khoongweihao
Version: 0.10
Last Updated: 17/05/2020
'''
__author__ = 'Wei Hao Khoong: https://github.com/weihao94'

import os

# models import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import lightgbm as lgb 
from xgboost import XGBRegressor, XGBClassifier
from xgboost import DMatrix

import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import euclidean, cosine, jaccard, chebyshev, correlation, cityblock, canberra, braycurtis, hamming
import random
from collections import namedtuple

class DEBoostRegressor:
    def __init__(self, method='regression', mode='mean', sdhw=True):
        self.type = method 
        self.mode = mode
        self.models = [Ridge(),
                       Lasso(),
                       ElasticNet(),
                       AdaBoostRegressor(),
                       GradientBoostingRegressor(),
                       RandomForestRegressor(),
                       SVR(),
                       DecisionTreeRegressor(),
                       lgb,
                       XGBRegressor()]
        self.sdhw = sdhw # smallest distance highest weight
    
    def fit(self, X, y):
        d_train = lgb.Dataset(X, label=y)
        self.models = [m.fit(X, y) for m in self.models if m != lgb] +\
                      [m.train({'verbose':-1}, d_train, verbose_eval=False) for m in self.models if m == lgb]
    
    def predict(self, X):
        # predictions is a list of 1D numpy arrays
        predictions = []
        for m in self.models:
            predictions += [m.predict(X)]
        if self.mode == 'mean':
            preds = self.get_mean_preds(predictions)
        if self.mode == 'median':
            preds = self.get_median_preds(predictions)
        if self.mode == 'dist_euclid':
            preds = self.get_dist_preds(predictions, 'euclid')
        if self.mode == 'dist_cosine':
            preds = self.get_dist_preds(predictions, 'cosine')
        if self.mode == 'dist_jaccard': # i think this is only for boolean
            preds = self.get_dist_preds(predictions, 'jaccard')
        if self.mode == 'dist_chebyshev':
            preds = self.get_dist_preds(predictions, 'chebyshev')
        if self.mode == 'dist_correlation':
            preds = self.get_dist_preds(predictions, 'correlation')
        if self.mode == 'dist_cityblock':
            preds = self.get_dist_preds(predictions, 'cityblock')
        if self.mode == 'dist_canberra':
            preds = self.get_dist_preds(predictions, 'canberra')
        if self.mode == 'dist_braycurtis':
            preds = self.get_dist_preds(predictions, 'braycurtis')
        if self.mode == 'dist_hamming': # i think this is only for boolean
            preds = self.get_dist_preds(predictions, 'hamming')
        if self.mode == 'dist_battacharyya':
            preds = self.get_dist_preds(predictions, 'battacharyya')

        return preds
    
    # function for getting mean of predictions
    def get_mean_preds(self, predictions):
        concat_preds = sum(predictions)
        return concat_preds/len(self.models)

    # function for getting median of predictions
    def get_median_preds(self, predictions):
        nrows = len(predictions[0])
        concat_df = pd.DataFrame({'id':list(range(nrows))})
        for i, pred in enumerate(predictions):
            concat_df['pred' + str(i)] = pred
        ncol = concat_df.shape[1]
        concat_df['median'] = concat_df.iloc[:, 1:ncol].median(axis=1)
        return concat_df[['median']].to_numpy()

    # function for getting weighted predictions with distance metric
    def get_dist_preds(self, predictions, metric):
        new_preds = []
        for j, pred in enumerate(predictions):
            distances = []
            remaining_preds = predictions[:j] + predictions[j+1:]
            for pred_ in remaining_preds:
                if metric == 'euclid':
                    distances += [euclidean(pred_, pred)]
                elif metric == 'cosine':
                    distances += [cosine(pred_, pred)]
                elif metric == 'jaccard': # i think this is only for boolean
                    distances += [jaccard(pred_, pred)]
                elif metric == 'chebyshev':
                    distances += [chebyshev(pred_, pred)]
                elif metric == 'correlation':
                    distances += [correlation(pred_, pred)]
                elif metric == 'cityblock':
                    distances += [cityblock(pred_, pred)]
                elif metric == 'canberra':
                    distances += [canberra(pred_, pred)]
                elif metric == 'braycurtis':
                    distances += [braycurtis(pred_, pred)]
                elif metric == 'hamming': # i think this is only for boolean
                    distances += [hamming(pred_, pred)]
                elif metric == 'battacharyya':
                    distances += [DistanceMetrics.battacharyya(pred_, pred, method='continuous')]
            new_preds += [(pred, sum(distances))] # (precdictions, weight)
        
        weights = [tup[1] for tup in new_preds]
        W = sum(weights) # total weight
        if self.sdhw:
            # those with lower distances have higher weight
            # sort in ascending order of aggregated distances
            preds_ascending_dist = sorted(new_preds, key=lambda x: x[1])
            weights_descending = sorted(weights, reverse=True)
            weighted_pred = sum([pred_tup[0]*(weights_descending[k]/W) for k, pred_tup in enumerate(preds_ascending_dist)])
        else:
            # those with lower distances have lower weight
            weighted_pred = sum([pred_tup[0]*(pred_tup[1]/W) for pred_tup in new_preds])
        return weighted_pred


class DEBoostClassifier:
    def __init__(self, method='regression', mode='mean', sdhw=True):
        self.type = method 
        self.mode = mode
        self.models = [AdaBoostClassifier(),
                       GradientBoostingClassifier(),
                       GaussianNB(),
                       KNeighborsClassifier(), 
                       LogisticRegression(), 
                       RandomForestClassifier(),
                       SVC(probability=True),
                       DecisionTreeClassifier(min_samples_leaf=31),
                       lgb.LGBMClassifier(),
                       XGBClassifier()]
        self.sdhw = sdhw # smallest distance highest weight
    
    def fit(self, X, y):
        self.models = [m.fit(X, y) for m in self.models]
    
    def predict(self, X): 
        # predictions is a list of 1D numpy arrays
        predictions = []
        for m in self.models:
            pred_tup = namedtuple('pred_tup', ['probabilities', 'classes'])
            predictions += [pred_tup(m.predict_proba(X), m.classes_)]
        if self.mode == 'mean':
            preds = self.get_mean_preds(predictions)
        if self.mode == 'median':
            preds = self.get_median_preds(predictions)
        if self.mode == 'dist_euclid':
            preds = self.get_dist_preds(predictions, 'euclid')
        if self.mode == 'dist_cosine':
            preds = self.get_dist_preds(predictions, 'cosine')
        if self.mode == 'dist_jaccard': # i think this is only for boolean
            preds = self.get_dist_preds(predictions, 'jaccard')
        if self.mode == 'dist_chebyshev':
            preds = self.get_dist_preds(predictions, 'chebyshev')
        if self.mode == 'dist_correlation':
            preds = self.get_dist_preds(predictions, 'correlation')
        if self.mode == 'dist_cityblock':
            preds = self.get_dist_preds(predictions, 'cityblock')
        if self.mode == 'dist_canberra':
            preds = self.get_dist_preds(predictions, 'canberra')
        if self.mode == 'dist_braycurtis':
            preds = self.get_dist_preds(predictions, 'braycurtis')
        if self.mode == 'dist_hamming': # i think this is only for boolean
            preds = self.get_dist_preds(predictions, 'hamming')
        if self.mode == 'dist_battacharyya':
            preds = self.get_dist_preds(predictions, 'battacharyya')
            
        return preds
    
    # function for getting mean of predictions
    def get_mean_preds(self, predictions):
        classes = predictions[0].classes
        preds = [pred.probabilities for pred in predictions]
        mean_probs = np.mean(preds, axis=0)
        indices_max_proba = mean_probs.argmax(axis=1)
        classifications = np.array([classes[i] for i in indices_max_proba])

        return classifications

    # function for getting median of predictions
    def get_median_preds(self, predictions):
        classes = predictions[0].classes
        preds = [pred.probabilities for pred in predictions]
        median_probs = np.median(preds, axis=0)
        indices_max_proba = median_probs.argmax(axis=1)
        classifications = np.array([classes[i] for i in indices_max_proba])

        return classifications

    # function for getting weighted predictions with distance metric
    def get_dist_preds(self, predictions, metric):
        new_preds = []
        classes = predictions[0].classes
        for j, pred in enumerate(predictions):
            distances = []
            remaining_preds = predictions[:j] + predictions[j+1:]
            for pred_ in remaining_preds:
                dist_by_class = list([0]*len(classes))
                for k, class_ in enumerate(classes):
                    class_pred_ = pred_.probabilities[k]
                    class_pred = pred.probabilities[k]
                    if metric == 'euclid':
                        dist_by_class[k] = euclidean(class_pred_, class_pred)
                    elif metric == 'cosine':
                        dist_by_class[k] = cosine(class_pred_, class_pred)
                    elif metric == 'jaccard': # i think this is only for boolean
                        dist_by_class[k] = jaccard(class_pred_, class_pred)
                    elif metric == 'chebyshev':
                        dist_by_class[k] = chebyshev(class_pred_, class_pred)
                    elif metric == 'correlation':
                        dist_by_class[k] = correlation(class_pred_, class_pred)
                    elif metric == 'cityblock':
                        dist_by_class[k] = cityblock(class_pred_, class_pred)
                    elif metric == 'canberra':
                        dist_by_class[k] = canberra(class_pred_, class_pred)
                    elif metric == 'braycurtis':
                        dist_by_class[k] = braycurtis(class_pred_, class_pred)
                    elif metric == 'hamming': # i think this is only for boolean
                        dist_by_class[k] = hamming(class_pred_, class_pred)
                    elif metric == 'battacharyya':
                        dist_by_class[k] = DistanceMetrics.battacharyya(class_pred_, class_pred, method='continuous')
                distances += [dist_by_class]
                # distances = [[c11,c21,c31], [c12,c22,c32], ..., [c1m,c2m,c3m]] for m models
            # new_preds = [(pred, [c11+...+c1m, ..., c31+...+c3m])]
            new_preds += [(pred, [sum(i) for i in zip(*distances)])] # (precdictions, [w1, w2, ..., wc]) for c classes
        
        weights = [tup[1] for tup in new_preds] 
        W = [sum(i) for i in zip(*weights)] # total weight for each class: [sum(w1i), sum(w2i), ..., sum(wci)], sum of sums for each model i
        class_weighted_preds = []
        for i, class_ in enumerate(classes):
            class_weights = [w[i] for w in weights]
            class_pred_dist = [(np.array([l[i] for l in tup[0].probabilities]), tup[1][i]) for tup in new_preds]
            if self.sdhw:
                # those with lower distances have higher weight sort in ascending order of aggregated distances
                preds_ascending_dist = sorted(class_pred_dist, key=lambda x: x[1])
                # weights is a list of lists containing the weights for the classes of each model
                weights_descending = sorted(class_weights, reverse=True)
                weighted_pred = sum([pred_tup[0]*(weights_descending[k]/W[i]) for k, pred_tup in enumerate(preds_ascending_dist)])
            else:
                # those with lower distances have lower weight
                weighted_pred = sum([pred_tup[0]*(pred_tup[1]/W[i]) for pred_tup in class_pred_dist])
            class_weighted_preds += [weighted_pred]
        class_weighted_preds_trunc = np.array([[class_weighted_preds[0][i], class_weighted_preds[1][i]] for i in range(len(class_weighted_preds[0]))])
        indices_max_proba = class_weighted_preds_trunc.argmax(axis=1)
        classifications = np.array([classes[i] for i in indices_max_proba])
        return classifications


class DistanceMetrics:
    '''
    - non-built-in distance metrics are found here
    - work in progress
    '''

    @staticmethod
    def get_density(x, cov_factor=0.1):
        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density
    
    @classmethod
    def battacharyya(cls, X1, X2, method='continuous'):
        '''
        Original Author: Eric Williamson (ericpaulwill@gmail.com)
        Obtained from: https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py

        - This calculates the Bhattacharyya distance between vectors X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
          feature in two separate classes.
        '''
        #Combine X1 and X2, we'll use it later:
        cX = np.concatenate((X1,X2))
        if method == 'noiseless':
            ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is
            ### treated as an individual bin.
            uX = np.unique(cX)
            A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
            A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
            bht = 0
            for x in uX:
                p1 = (X1==x).sum() / A1
                p2 = (X2==x).sum() / A2
                bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

        elif method == 'hist':
            ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)
            N_BINS = int(len(X1) * 2)
            #Bin the values:
            h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
            h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
            #Calc coeff from bin densities:
            bht = 0
            for i in range(N_BINS):
                p1 = h1[i]
                p2 = h2[i]
                bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

        elif method == 'autohist':
            ###Bin the values into bins automatically set by np.histogram:
            #Create bins from the combined sets:
            # bins = np.histogram(cX, bins='fd')[1]
            bins = np.histogram(cX, bins='doane')[1] #Seems to work best
            # bins = np.histogram(cX, bins='auto')[1]

            h1 = np.histogram(X1,bins=bins, density=True)[0]
            h2 = np.histogram(X2,bins=bins, density=True)[0]

            #Calc coeff from bin densities:
            bht = 0
            for i in range(len(h1)):
                p1 = h1[i]
                p2 = h2[i]
                bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

        elif method == 'continuous':
            ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
            N_STEPS = int(len(X1) * 20)
            #Get density functions:
            d1 = cls.get_density(X1)
            d2 = cls.get_density(X2)
            #Calc coeff:
            xs = np.linspace(min(cX),max(cX),N_STEPS)
            bht = 0
            for x in xs:
                p1 = d1(x)
                p2 = d2(x)
                bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS

        else:
            raise ValueError("The value of the 'method' parameter does not match any known method")

        ###Lastly, convert the coefficient into distance:
        if bht==0:
            return float('Inf')
        else:
            return -np.log(bht)