'''
Created on Oct 22, 2017

@author: krish.mahajan
'''
import numpy as np



def tokens_to_vector_train(tokens,word_index_map,label_dict): 
    '''
    Vecotrizing particular token in test/train as per 
    Hashmap created Above          
    '''
    x = np.zeros(len(word_index_map)+1) 
    for t in tokens[:-1]: 
            j = word_index_map[t]
            x[j] +=1  
    x[-1]=label_dict[tokens[-1]]

    return x 

def tokens_to_vector_test(tokens,word_index_map): 
    '''
    Vecotrizing particular token in test/train as per 
    Hashmap created Above          
    '''
    x = np.zeros(len(word_index_map)) 
    for t in tokens[:]: 
            j = word_index_map[t]
            x[j] +=1  
    return x 

def vectorize_train_data(data,word_index_map,tokenized):  
    """
    Vectorizing all the text in training and testing
    """
    label_dict={}
    i=0
    for label in data['label'].unique():
        label_dict[label]=i
        i+=1
    N = len(tokenized)-1
    data_vector = np.zeros((N,len(word_index_map)+1)) 
    i=0
    for tokens in tokenized[1:]:
        xy = tokens_to_vector_train(tokens,word_index_map,label_dict)   
        data_vector[i,:] = xy 
        i +=1    
 
    return data_vector,label_dict 


def vectorize_test_data(data,word_index_map,tokenized):  
    """
    Vectorizing all the text in training and testing
    """

    N = len(tokenized)
    data_vector = np.zeros((N,len(word_index_map))) 
    i=0
    for tokens in tokenized[1:]:
        xy = tokens_to_vector_test(tokens,word_index_map)   
        data_vector[i,:] = xy 
        i +=1    
 
    return data_vector

