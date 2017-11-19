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
    unlabel_directory = sys.argv[3] 
    
    
    if(mode=="train"):  
        
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


    if(mode=="test"): 

        
        print("reading stopwords") 
        stopwords = read_stopwords(path=None)
        
        print("opening naieve bayes model for label model") 
        ff = open("./output/model-file_label.p","r")
        nb_label=pickle.load(ff)
        ff.close() 
 
                
        print("opening Saved semi supervised naieve bayes model for label + unlabel model") 
        ff = open("./output/model-file_unlabel.p","r")
        nb_unlabel=pickle.load(ff)
        ff.close() 


        print("Reading Test Data...")
        test_data=read_data_test(label_directory)   
    
        print("Tokinizing Test Data for label model")
        testing_tokenized_label =tokenize_test_data(test_data,nb_label.vocab,3,stopwords)    


        print("Tokinizing Test Data for label+ unlabel model")
        testing_tokenized_unlabel =tokenize_test_data(test_data,nb_unlabel.vocab,3,stopwords)    


        print("Vectorizing Test Data for label model...")
        word_index_map_label=nb_label.vocab
        label_dict_label = nb_label.label_dict
        test_data_vector_label= vectorize_test_data(test_data,word_index_map_label,testing_tokenized_label)


        print("Vectorizing Test Data for label+ unlabel model...")
        word_index_map_unlabel=nb_unlabel.vocab
        label_dict_unlabel = nb_unlabel.label_dict
        test_data_vector_unlabel= vectorize_test_data(test_data,word_index_map_unlabel,testing_tokenized_unlabel)



        print("Making term document matrix of testing label model")
        tf_test_label = term_document(test_data_vector_label) 

        print("Making term document matrix of testing label+unlabel model")
        tf_test_unlabel = term_document(test_data_vector_unlabel) 

    
        print("Predicting test Data for label model")
        predict_label=nb_label.predict_proba_all(tf_test_label)  
        
        print("Predicting test Data for label+unlabel model")
        predict_unlabel=nb_unlabel.predict_proba_all(tf_test_unlabel) 

        print("Making predictions for label and label+unlabel model")
        df_predictions_label=nb_label.print_predictions(predict_label, test_data, label_dict_label,path="./output/predictions_label.csv")
        df_predictions_unlabel=nb_unlabel.print_predictions(predict_unlabel, test_data, label_dict_unlabel,path="./output/predictions_unlabel.csv")

     
        print("calculating accuracy for label model")
        df_actual = read_data_accuracy("../data/testResult")


        accuracy_label, confusion_matrix_label = nb_label.accuracy(df_predictions_label,df_actual,path='./output/predictions_comparison_label.csv') 
        accuracy_unlabel, confusion_matrix_unlabel = nb_label.accuracy(df_predictions_unlabel,df_actual,path='./output/predictions_comparison_unlabel.csv')
        
        print("Accuracy for label model is",accuracy_label) 
        print("Accuracy for label+ unlabel model is",accuracy_unlabel)

        print("Confusion Matrix label model:") 
        print(confusion_matrix_label)
        
        print("Confusion Matrix label+unlabel model:") 
        print(confusion_matrix_unlabel)
     
    
    