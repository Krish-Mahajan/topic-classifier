'''
Created on Oct 16, 2017

@author: krish.mahajan
'''
from sklearn.metrics.classification import accuracy_score
"""
Text Classification using Naive Bayes on Expectation Maximization Principle
w: word
x: document
T: topic/class
V: vocabulary 
M: number of documents
T: number of classes
""" 

import numpy as np
import pandas as pd
import pickle

class NaiveBayes(object):

    def __init__(self, vocab=None,label_dict=None):

        self.p_w_c = None
        self.p_c = None
        self.vocab=vocab
        self.label_dict=label_dict   

        # to normalize p(c)
    def normalize_p_c(self,p_c):
        M = len(p_c)
        denom = M + np.sum(p_c)
        p_c += 1.0
        p_c /= denom
        p_c /=np.sum(p_c)
        
        # to normalize p(w|c)
    def normalize_p_w_c(self,p_w_c):
        V, X = p_w_c.shape
        denoms = V + np.sum(p_w_c, axis=0)
        p_w_c += 1.0
        p_w_c /= denoms[np.newaxis,:]  


    def top_10_words(self,name,path): 
        print(path)
        if not path:f = open("./output/distinctive_words_"+name+ ".txt","w") 
        else: f = open(path,"w") 
        for i in range(self.p_w_c.shape[1]):
            column = []
            for x,y in np.ndenumerate(self.p_w_c[:,i]):
                column.append((x[0],y)) 
            top_10 = sorted(column, key=lambda x: x[1])[-10:] 
            top_10.reverse()
            f.write("Top words for topic  "+ self.label_dict.keys()[self.label_dict.values().index(i)].upper())
            f.write("\n")
            for word in top_10:
                f.write(self.vocab.keys()[self.vocab.values().index(word[0])])
                f.write("\n") 
            f.write("\n")
        f.close() 
        return f


    def save_model(self,name,path): 
        if not path: f = open("./output/model-file_"+ name + ".p","w")       
        else : f =  open(path,"w")
        pickle.dump(self,f)
        f.close()

        
    # to train naive bayes model
    def train(self, td, delta, normalize=True):

        X_, M = delta.shape
        V, X = td.shape

        # P(c)
        self.p_c = np.sum(delta, axis=0)

        # P(w|c)
        self.p_w_c = np.zeros((V,M), dtype=np.double)

        for w,d in zip(*td.nonzero()):
            self.p_w_c[w,:] += td[w,d] * delta[d,:]


        if normalize:
            self.normalize_p_c(self.p_c)
            self.normalize_p_w_c(self.p_w_c) 

        #to train naive bayes when model is semi supervised
    def train_semi(self, td, delta, tdu, maxiter=5):
        X_, M = delta.shape
        V, X = td.shape

        # compute counts for labeled data once for all
        self.train(td, delta, normalize=False)
        p_c_l = np.array(self.p_c, copy=True)
        p_w_c_l = np.array(self.p_w_c, copy=True)

        # normalize to get initial classifier
        self.normalize_p_c(self.p_c)
        self.normalize_p_w_c(self.p_w_c)

        for iteration in range(1, maxiter+1):
            # E-step: 
            print("iteration no ",iteration," out of total",maxiter)
            delta_u = self.predict_proba_all(tdu)

            # M-step: 
            self.train(tdu, delta_u, normalize=False)
            self.p_c += p_c_l
            self.p_w_c += p_w_c_l
            self.normalize_p_c(self.p_c)
            self.normalize_p_w_c(self.p_w_c)

    def p_x_c_log_all(self, td):
        M = len(self.p_c)
        V, X = td.shape
        p_x_c_log = np.zeros((X,M), np.double)
        p_w_c_log = np.log(self.p_w_c)

        for w,d in zip(*td.nonzero()):
            p_x_c_log[d,:] += p_w_c_log[w,:] * td[w,d]

        return p_x_c_log

    def max_prob(self,loga, k=-np.inf, out=None):
        if out is None: out = np.empty_like(loga).astype(np.double)
        m = np.max(loga)
        logam = loga - m
        sup = logam > k
        inf = np.logical_not(sup)
        out[sup] = np.exp(logam[sup])
        out[inf] = 0.0
        out /= np.sum(out)
        return out

    def predict_proba_all(self, td):
        V, X = td.shape
        p_x_c_log = self.p_x_c_log_all(td)
        p_x_c_log += np.log(self.p_c)[np.newaxis,:]
        for d in range(X):
            self.max_prob(p_x_c_log[d,:], k=-10, out=p_x_c_log[d,:]) 

        return p_x_c_log

        
    
    def accuracy(self,y_pred,y_actual,path=None): 
        if not path : path = './output/predictions_comparison.csv' 
        
        i=0 
        df_compare_cols =['id','actual','predicted']
        df_compare = []
        
        for index1,row1 in y_actual.iterrows(): 
            index2= y_pred.index[y_pred['id']==row1['id']].tolist()[0] 
            df_compare.append([y_pred.iloc[index2]['id'],row1['topic'],y_pred.iloc[index2]['predicted']])
            if(y_pred.iloc[index2]['predicted']==row1['topic']):
                i+=1  
                
        df = pd.DataFrame(df_compare,columns=df_compare_cols) 
        df_confusion= pd.crosstab(df.ix[:,1],df.ix[:,2])
        
        accuracy = i*100.0/len(y_pred) 
        
        df.to_csv(path, header=['id','actual','predicted'], index=None, sep=',', mode='a')
        return accuracy,df_confusion
                


    
       
    def print_predictions(self,predict,test_data,label_dict,path=None): 
        if not path : path = './output/predictions.csv'
        y_pred=[np.argmax(row) for row in predict]
        
        for row in range(len(y_pred)):
            for key,value in label_dict.iteritems():
                if value==y_pred[row]: y_pred[row]=key 

        y_pred = pd.Series(y_pred,name='predicted')  
        y_filename = test_data.ix[:,0] 
    
 
        y_df = pd.concat([y_filename,y_pred],axis=1) 
        y_df.sort_values(by='id')
        y_df.to_csv(path, header=['fileno','predicted_topic'], index=None, sep=',', mode='a') 
        
        return y_df
        

