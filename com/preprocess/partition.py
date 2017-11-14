'''
Created on Oct 22, 2017

@author: krish.mahajan
'''


import random
import numpy as np




def partition_label_unlabeled(train_data_vector,label_dict):
    """
    To segregate training data into labeled and non labeled based on fraction
    """ 


    for index in range(len(train_data_vector)):
        if ("unknown" in label_dict.keys()):
            if (train_data_vector[index][-1]==label_dict["unknown"]):train_data_vector[index][-1]=-1
    train_data_vector_labeled = train_data_vector[train_data_vector[:,-1]!=-1]
    train_data_vector_unlabeled = train_data_vector[train_data_vector[:,-1]==-1]
    
    return train_data_vector_labeled,train_data_vector_unlabeled,label_dict


def coin_flip(fraction):
    return (random.random()>=1-fraction)


def partition_label_unlabeled_old(train_data_vector,fraction):
    """
    To segregate training data into labeled and non labeled based on fraction
    """ 
    if fraction ==0:
        fraction = 0.0001

    for index in range(len(train_data_vector)):
        if (not coin_flip(fraction)):train_data_vector[index][-1]=-1
    train_data_vector_labeled = train_data_vector[train_data_vector[:,-1]!=-1]
    train_data_vector_unlabeled = train_data_vector[train_data_vector[:,-1]==-1]
    
    print(len(train_data_vector_labeled))
    print(len(train_data_vector_unlabeled))
    return train_data_vector_labeled,train_data_vector_unlabeled



def term_document(data): 
    """
    td: term-document matrix V x D
    """
    td = data.T[:-1,:] 
    return td 


def document_class(data,td):
    """
    delta: D x T matrix
    where delta_train(d,c) = 1 if document d belongs to class c
    """
    total_classes = len(np.unique(data[:,-1]))
    total_documents = td.shape[1]
    document_class = np.zeros((total_documents,total_classes))  
    
    label=data[:,-1]  

    #Filling Delta
    for row in range(len(label)) :
        document_class[row,int(label[row])]=1
        
    return document_class 

def document_label(data,td):
    """
    delta: D x T matrix
    where delta_train(d,c) = 1 if document d belongs to class c
    """
    total_classes = len(np.unique(data[:,-1]))
    total_documents = td.shape[1]
    delta_train = np.zeros((total_documents,total_classes))  
    
    label=data[:,-1]  

    #Filling Delta
    for row in range(len(label)) :
        delta_train[row,int(label[row])]=1
        
    return delta_train
