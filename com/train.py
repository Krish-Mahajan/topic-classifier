'''
Created on Nov 19, 2017

@author: krish.mahajan 
'''  

import pickle  
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from com.preprocess.readdata import *
from com.preprocess.tokenizer import tokenize_train_data
from com.preprocess.vectorize import *
from com.preprocess.partition import *
from com.preprocess.logic import snb
from com.preprocess.tokenizer import tokenize_test_data


def solve_train(label_directory,unlabel_directory):
    print("reading stopwords") 
    stopwords = read_stopwords(path=None)

    print("Reading Training Data...") 
    if unlabel_directory != 'NOT' : train_data = read_data(label_directory,unlabel_directory)
    else : train_data = read_data(label_directory)
    
    print("First 10 rows of train data")
    print(train_data.head(10)) 
    
    
    print("Tokinizing Train Data and Creating Vocabularly")
    train_data,word_index_map,training_tokenized=tokenize_train_data(train_data,3,stopwords)   


    print("Vectorizing Training Data...")
    train_data_vector,label_dict = vectorize_train_data(train_data,word_index_map,training_tokenized)

    #print("Splitting training Data into label and unlabeled as per ",fraction,"probability of looking at each label")
    train_data_vector_label,train_data_vector_unlabel,label_dict = partition_label_unlabeled(train_data_vector,label_dict)
    #train_data_vector_label,train_data_vector_unlabel = partition_label_unlabeled_old(train_data_vector,fraction)
    label_dict.pop('unknown', None) 
    
    print("Making term document matrix of Training label Data and Training unlabel Data")
    tf_train_label = term_document(train_data_vector_label) 
    tf_train_unlabel = term_document(train_data_vector_unlabel) 

    print("Making document_class  on training label")
    document_class_train_label = document_class(train_data_vector_label,tf_train_label) 

    print("Data Prepared")  

    print("Making Expectation Maximization Naieve Bayes object for label and label + unlabel data ")
    nb_label=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict)
    nb_unlabel=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict) 
    
    print("Training naive Bayes Model on just label data") 
    nb_label.train(tf_train_label,document_class_train_label)
    
    
    print("Training EM naive Bayes Model on label + unlabel data") 
    nb_unlabel.train_semi(tf_train_label,document_class_train_label,tf_train_unlabel,maxiter=10)  


    print("Making distinctive_words  top 10 words in each topic for label and label+ unlabel data")
    nb_label.top_10_words("label",path=None)
    nb_unlabel.top_10_words("unlabel",path=None)

    print("Saving Trained Models..")
    nb_label.save_model("label",path=None) 
    nb_unlabel.save_model("unlabel",path=None)
