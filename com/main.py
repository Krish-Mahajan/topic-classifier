'''
Created on Oct 22, 2017

@author: krish.mahajan
'''

import warnings 
import sys 
import os 
from train import *
from test import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__=="__main__": 
    
    warnings.filterwarnings('ignore')
    mode = sys.argv[1]  
    label_directory = sys.argv[2] 
    unlabel_directory = sys.argv[3] 
      
    if(mode=="train"):  
        solve_train(label_directory,unlabel_directory)
 

    if(mode=="test"): 
        solve_test(label_directory,unlabel_directory)
 
    
    