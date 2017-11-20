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


def solve_test(label_directory,unlabel_directory):
           
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
     