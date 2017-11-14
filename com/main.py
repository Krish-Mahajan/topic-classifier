'''
Created on Oct 22, 2017

@author: krish.mahajan
'''

import warnings 
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle 
from com.preprocess.readdata import *
from com.preprocess.tokenizer import tokenize_train_data
from com.preprocess.vectorize import *
from com.preprocess.partition import *
from com.preprocess.logic import snb
from com.preprocess.tokenizer import tokenize_test_data


if __name__=="__main__": 
    
    warnings.filterwarnings('ignore')
    mode = sys.argv[1] 
    label_directory = sys.argv[2] 
    model_file = sys.argv[3]
    fraction = float(sys.argv[4]) 
    unlabel_directory = sys.argv[5] 
    
    
    if(mode=="train"):  
        
        print("reading stopwords") 
        stopwords = read_stopwords(path=None)

        print("Reading Training Data...") 
        if unlabel_directory != 'NOT' : train_data = read_data(label_directory,unlabel_directory)
        else : train_data = read_data(label_directory)
        
        print(train_data.head(10))
        print("Tokinizing Train Data and Creating Vocabularly")
        train_data,word_index_map,training_tokenized=tokenize_train_data(train_data,3,stopwords)   


        print("Vectorizing Training Data...")
        train_data_vector,label_dict = vectorize_train_data(train_data,word_index_map,training_tokenized)

        print("Splitting training Data into label and unlabeled as per ",fraction,"probability of looking at each label")
        
        #train_data_vector_label,train_data_vector_unlabel,label_dict = partition_label_unlabeled(train_data_vector,label_dict)
        train_data_vector_label,train_data_vector_unlabel = partition_label_unlabeled_old(train_data_vector,fraction)

        print("Making term document matrix of Training label Data and Training unlabel Data")
        tf_train_label = term_document(train_data_vector_label) 
        tf_train_unlabel = term_document(train_data_vector_unlabel) 

        print("Making Delta on training label")
        document_class_train_label = document_class(train_data_vector_label,tf_train_label) 

        print("Data Prepared")  

        nb_label=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict)
        nb_unlabel=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict) 
        
        print("Training pure naive Bayes Model on just label data") 
        nb_label.train(tf_train_label,document_class_train_label)
        
        
        print("Training EM naive Bayes Model on label + unlabel data") 
        nb_unlabel.train_semi(tf_train_label,document_class_train_label,tf_train_unlabel,maxiter=10)  


        print("Making distinctive_words for top 10 words in each topic")
        nb_label.top_10_words("label",path=None)
        nb_unlabel.top_10_words("unlabel",path=None)

        print("Saving Trained Model..")
        nb_label.save_model("label",path=None) 
        nb_unlabel.save_model("unlabel",path=None)


    if(mode=="test"): 

        
        print("reading stopwords") 
        stopwords = read_stopwords(path=None)
        
                
        print("opening Saved semi supervised naieve bayes model") 
        ff = open("./output/model-file_unlabel.p","r")
        nb=pickle.load(ff)
        ff.close() 

        print("Reading Testing Data...")
        test_data=read_data_test(label_directory)   
    
        

        print("Tokinizing Test Data")
        testing_tokenized =tokenize_test_data(test_data,nb.vocab,3,stopwords)    


        print("Vectorizing Test Data...")
        word_index_map=nb.vocab
        label_dict = nb.label_dict
        test_data_vector= vectorize_test_data(test_data,word_index_map,testing_tokenized)

        print("Making term document matrix of testing Data")
        tf_test = term_document(test_data_vector) 

        
    
        print("Predicting test Data")
        predict=nb.predict_proba_all(tf_test) 
        
        df_predictions=nb.print_predictions(predict, test_data, label_dict,path=None)

     
        print("calculating accuracy")
        df_actual = read_data_accuracy("../data/testResult")


        accuracy, confusion_matrix = nb.accuracy(df_predictions,df_actual,path=None)
        
        print("Accuracy is",accuracy)

        print("Confusion Matrix:") 
        #print(confusion_matrix)
        
     
    
    