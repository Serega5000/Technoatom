import pandas as pd
import numpy as np
import matplotlib.pyplot as pltv
from scipy import stats
import numpy.ma as ma

def accuracy_score(y_true, y_predict, percent):
    if  (percent==None):
        r=np.round(y_predict)
        a=np.greater(r[:,0],r[:,1])
        x1=np.zeros(len(y_true))
        x2=np.ones(len(y_true))
        y=np.where(a,x1,x2)
        return np.mean(y==y_true)
    if (percent>=1)&(percent<=100):
        d=round((len(y_true)/100)*percent)
        g=y_predict[:,1]
        g1=y_true[np.argsort(g)]
        g2=g1[::-1]
        g3=g[np.argsort(g)]
        g4=g3[::-1]
        y_true_n=g2[:d]
        y_predict_n=np.round(g4[:d])
        return np.mean(y_predict_n==y_true_n)
    else:
        print ('error: percent belong [1,100]')

def precision_score(y_true, y_predict, percent=None):
    if  (percent==None):
        r=np.round(y_predict)
        a=np.greater(r[:,0],r[:,1])
        x1=np.zeros(len(y_true))
        x2=np.ones(len(y_true))
        y=np.where(a,x1,x2)
        TP=np.sum((y_true==1)&(y==1))
        FP=np.sum((y_true==0)&(y==1))
        return TP/(TP+FP)
    if (percent>=1)&(percent<=100):
        d=round((len(y_true)/100)*percent)
        g=y_predict[:,1]
        g1=y_true[np.argsort(g)]
        g2=g1[::-1]
        g3=g[np.argsort(g)]
        g4=g3[::-1]
        y_true_n=g2[:d]
        y_predict_n=np.round(g4[:d])  
        TP=np.sum((y_true_n==1)&(y_predict_n==1))
        FP=np.sum((y_true_n==0)&(y_predict_n==1))
        return TP/(TP+FP)
    else:
        print ('error: percent belong [1,100]')

def recall_score(y_true, y_predict, percent=None):
    if  (percent==None):
        r=np.round(y_predict)
        a=np.greater(r[:,0],r[:,1])
        x1=np.zeros(len(y_true))
        x2=np.ones(len(y_true))
        y=np.where(a,x1,x2)
        TP=np.sum((y_true==1)&(y==1))
        FN=np.sum((y_true==1)&(y==0))
        return TP/(TP+FN)
    if (percent>=1)&(percent<=100):
        d=round((len(y_true)/100)*percent)
        g=y_predict[:,1]
        g1=y_true[np.argsort(g)]
        g2=g1[::-1]
        g3=g[np.argsort(g)]
        g4=g3[::-1]
        y_true_n=g2[:d]
        y_predict_n=np.round(g4[:d])  
        TP=np.sum((y_true_n==1)&(y_predict_n==1))
        FN=np.sum((y_true_n==1)&(y_predict_n==0))
        return TP/(TP+FN)
    else:
        print ('error: percent belong [1,100]')

def lift_score(y_true, y_predict, percent=None):
    if  (percent==None):
        r=np.round(y_predict)
        a=np.greater(r[:,0],r[:,1])
        x1=np.zeros(len(y_true))
        x2=np.ones(len(y_true))
        y=np.where(a,x1,x2)
        TP=np.sum((y_true==1)&(y==1))
        FN=np.sum((y_true==1)&(y==0))
        FP=np.sum((y_true==0)&(y==1))
        precision=TP/(TP+FP)
        l=len(y_true)
        return  (precision/(TP+FN))/l
    if (percent>=1)&(percent<=100):
        d=round((len(y_true)/100)*percent)
        g=y_predict[:,1]
        g1=y_true[np.argsort(g)]
        g2=g1[::-1]
        g3=g[np.argsort(g)]
        g4=g3[::-1]
        y_true_n=g2[:d]
        y_predict_n=np.round(g4[:d])  
        TP=np.sum((y_true_n==1)&(y_predict_n==1))
        FN=np.sum((y_true_n==1)&(y_predict_n==0))
        FP=np.sum((y_true_n==0)&(y_predict_n==1))
        precision=TP/(TP+FP)
        l=len(y_predict_n)
        return (precision/(TP+FN))/l
    else:
        print ('error: percent belong [1,100]')

def f1_score(y_true, y_predict, b,percent=None):
    if  (percent==None):
        r=np.round(y_predict)
        a=np.greater(r[:,0],r[:,1])
        x1=np.zeros(len(y_true))
        x2=np.ones(len(y_true))
        y=np.where(a,x1,x2)
        TP=np.sum((y_true==1)&(y==1))
        FN=np.sum((y_true==1)&(y==0))
        FP=np.sum((y_true==0)&(y==1))
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        return ((precision*recall)*(1+b**2))/(b**2*precision+recall)
    if (percent>=1)&(percent<=100):
        d=round((len(y_true)/100)*percent)
        g=y_predict[:,1]
        g1=y_true[np.argsort(g)]
        g2=g1[::-1]
        g3=g[np.argsort(g)]
        g4=g3[::-1]
        y_true_n=g2[:d]
        y_predict_n=np.round(g4[:d])  
        TP=np.sum((y_true_n==1)&(y_predict_n==1))
        FN=np.sum((y_true_n==1)&(y_predict_n==0))
        FP=np.sum((y_true_n==0)&(y_predict_n==1))
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        return ((precision*recall)*(1+b**2))/(b**2*precision+recall)
    else:
        print ('error: percent belong [1,100]')