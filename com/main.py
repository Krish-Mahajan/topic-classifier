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
from com.preprocess.partition import partition_label_unlabeled
from com.preprocess.partition import term_document
from com.preprocess.partition import delta 
from com.preprocess.logic import snb
from com.preprocess.tokenizer import tokenize_test_data


if __name__=="__main__": 
    
    warnings.filterwarnings('ignore')
    mode = sys.argv[1] 
    dataset_directory = sys.argv[2] 
    model_file = sys.argv[3]
    fraction = float(sys.argv[4]) 
 
    if(mode=="train"):  
        
        print("reading stopwords") 
        stopwords = read_stopwords(path=None)

        print("Reading Training Data...")
        train_data = read_data(dataset_directory)  
        
        print(train_data.head(10))
        print("Tokinizing Train Data and Creating Vocabularly")
        train_data,word_index_map,training_tokenized=tokenize_train_data(train_data,3,stopwords)   


        print("Vectorizing Training Data...")
        train_data_vector,label_dict = vectorize_train_data(train_data,word_index_map,training_tokenized)

        print("Splitting training Data into label and unlabeled as per ",fraction,"probability of looking at each label")
        train_data_vector_label,train_data_vector_unlabel = partition_label_unlabeled(train_data_vector,fraction)


        print("Making term document matrix of Training label Data and Training unlabel Data")
        tf_train_label = term_document(train_data_vector_label) 
        tf_train_unlabel = term_document(train_data_vector_unlabel) 

        print("Making Delta on training label")
        delta_train_label = delta(train_data_vector_label,tf_train_label) 

        print("Data Prepared")  

        nb=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict)

        print("Training EM naive Bayes Model...") 
        nb.train_semi(tf_train_label,delta_train_label,tf_train_unlabel)  


        print("Making distinctive_words for top 10 words in each topic")
        nb.top_10_words(path=None)


        print("Saving Trained Model..")
        nb.save_model(path=None)


    if(mode=="test"): 

        
        print("reading stopwords") 
        stopwords = read_stopwords(path=None)
        
        print("opening Saved semi supervised naieve bayes model") 
        ff = open("./output/model-file.p","r")
        nb=pickle.load(ff)
        ff.close() 

        print("Reading Testing Data...")
        test_data=read_data_test(dataset_directory)   
        print(test_data.shape)
        

        print("Tokinizing Test Data")
        testing_tokenized =tokenize_test_data(test_data,nb.vocab,3,stopwords)    


        print("Vectorizing Test Data...")
        word_index_map=nb.vocab
        label_dict = nb.label_dict
        test_data_vector= vectorize_test_data(test_data,word_index_map,testing_tokenized)

        print("Making term document matrix of testing Data")
        tf_test = term_document(test_data_vector) 

        '''
        print("Making Delta on testing label")
        delta_test = delta(test_data_vector,tf_test) 
        '''
        
    
        print("Predicting test Data")
        predict=nb.predict_proba_all(tf_test) 
        
        df_predictions=nb.print_predictions(predict, test_data, label_dict)
        print(df_predictions.head())
     
        print("calculating accuracy")
        df_actual = read_data_accuracy("../data/testResult")
        print(df_actual.head()) 
        print(df_actual.shape)
        print(df_actual[df_actual.id=='54626']) 
        print(df_predictions[df_predictions.id=='54626']) 
        
        
        print("Accuracy is",nb.accuracy_new(df_predictions,df_actual))

        print("Confusion Matrix:")
        #print(nb.confusion_matrix(delta_test,predict,nb.label_dict))
        
     
    
    