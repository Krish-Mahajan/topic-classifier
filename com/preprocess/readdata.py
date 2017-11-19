'''
Created on Oct 22, 2017

@author: krish.mahajan
'''

import os
import pandas as pd 




def read_stopwords(path=None): 
    '''
    function to read stopwords which are predetermined earlier
    '''
    if not path  : path = './resources/stopwords.txt'
    stopwords= set()
    for line in open(path): 
        for words in line.split(','): 
            stopwords.add(words) 
    return stopwords 



## Function to prepare training & testing data in tabular form 
def read_data(trainlabel,trainunlabel=None):
    """
    To read all the training data and return data as pandas data frame
    with three columns : id,text,label
    """ 

    df_label = pd.read_csv(trainlabel) 
    print("Shape of label data")
    print(df_label.shape)
    df_combine = df_label
    if trainunlabel :
        df_unlabel = pd.read_csv(trainunlabel) 
        print("Shape of unlabel data")
        print(df_unlabel.shape)
        df_combine = pd.concat([df_combine,df_unlabel],axis=0) 
    df_combine = df_combine.dropna() 
    print("shape of label + unlabel data")
    print(df_combine.shape) 
    df_combine = df_combine[["id","label","text"]]
    #df_combine.to_csv('/home/krish.mahajan/Documents/other_projects/topic-classification/TopicClassfier/data/train/df_combine.csv',sep=',')
    return df_combine


## Function to prepare training & testing data in tabular form 
def read_data_accuracy(path):
    """
    To read all the training data and return data as pandas data frame
    with three columns : id,text,label
    """ 
    data = []
    for topic in os.listdir(path):  
        if (topic != ".DS_Store"):
            new_path = "./"+path +"/" + topic 
            for document in os.listdir(new_path): 
                fo = open(new_path + "/" + document, 'r')
                content = fo.read()
                fo.close 
                data.append({'id': str(document),'topic':topic})
    df = pd.DataFrame(data)
    df.sort_values(by='id')
    return df



def read_data_test(path):
    """
    To read all the training data and return data as pandas data frame
    with three columns : id,text,label
    """
    data = []
    i=0
    for topic in os.listdir(path):  
        i+=1
        if (topic != ".DS_Store"):
            new_path = "./"+path +"/" + topic 
            fo = open(new_path, 'r')
            content = fo.read()
            fo.close 
            data.append({'id': str(topic),'text':content}) 

    return pd.DataFrame(data)