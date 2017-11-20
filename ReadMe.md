



Topic Classifier
-----------------------

This projects uses principles of semi-supervised text classification using Expectation Maximization algorithm to predict the topics of test dataset using training datasets which are labeled as well as unlabeled.   
Label datasets includes the topic information about the text whereas unlabel data just has text data. 
In this project i have implemented [expectation maximization naieve bayes](https://www.cs.cmu.edu/~tom/pubs/NigamEtAl-bookChapter.pdf) algorithm to train on the label datasets as well as leverage unlabel datasets. 
 

Folder Structure 
============================

### top-level directory layout

    .
    |-- data                        # required training and testing data
    |-- com/preprocess/logic/*      # contains source files of semi supervised EM Algorithm
    |--  com/preprocess/*           # contains source files of python module for preprocessing data
    |-- com/resources               # contains resources required by the project
    |-- requirements.txt            # contains required Tools and utilities 
    |--  main.py					# python main module
    |-- train.py					# Module to train dataset 
    |-- test.py 					# Module to make predction on test dataset 
    |-- ipython.ipynb				# Step by Step tutorial about the project through Jupyter notebook
    |-- LICENSE
    |-- README.md

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 2.7.
    * You may want to use a virtual environment for this.

Usage
-----------------------

* Run `python main.py train <label_data_file> <unlabel_data_file>` to train the algorithm for label + unlabel dataset 
	* This will create `distinctive_words.txt` and `model_file.p` in the `com/output` folder.
* Run `python main.py test <test_data_file>  x` to predict the topics of test dataset  a
    * This will create `predictions.csv` in the `com/output` folder. 
    * This will also output accuracy on training dataset when data was trained on just label data as well as label+unlabel data  
    * This module 'll also output confusion matrix



# Results     

* For dataset with label data = 543 rows and unlabel data = 11314 rows  with 20 possible classes
	* Accuracy = 54% when trained on just label data  and test data size = 6870 rows
	* Accuracy = 62% when trained on label + unlabel data and test data size = 6870 rows    
 



