
*Work In Progress*

Directorty structure  
--------------------- 

training/topic1/documents.....     
training/topic2/documents.....    
training/topic3/documents.....    
training/topic4/documents....  
training/topic5/documents...  
training/topic6/documents....       
training/topic_n/documents....    



testing/topic1/documents.....     
testing/topic2/documents.....    
testing/topic3/documents.....    
testing/topic4/documents....  
testing/topic5/documents...  
testing/topic6/documents....       
testing/topic_n/documents....    


topics.py  
snb.py   
readme.md  

### Instruction to train the data. (for ex.)    

python main.py train /home/krish.mahajan/Documents/other_projects/topic-classification/TopicClassfier/data/train/train.csv model-file 0.7 NOT #From Command Line old method 

python main.py train /home/krish.mahajan/Documents/other_projects/topic-classification/TopicClassfier/data/train_labeled/trainlabel.csv model-file 1.0  /home/krish.mahajan/Documents/other_projects/topic-classification/TopicClassfier/data/train_unlabeled/trainunlabel.csv    ## From Command Line new Method
 
python main.py train ../data/train_labeled model-file 1.0  ##from command line
      mode = train   
      dataset-directory = training   
      model-file = model-file   
      fraction = 0.8 (Could be any number between [0,1])  
      train data/train model-file 0.8

### Instruction to test the data. (for ex.)  
python main.py test ../data/test  model-file 1.0  /home/krish.mahajan/Documents/other_projects/topic-classification/TopicClassfier/data/train_unlabeled/trainunlabel.csv  ##from command line
      mode = test  
      dataset-directory = testing  
      model-file = model-file   
      fraction = same as fraction while train mode  
      test data/test model-file.p 0.8



# Results     

- For fraction 1 :Accuracy = 82%      
- For fraction 0.5 : Accuracy 79%    
- For Fraction 0.01 : Accuracy 42%     
- For fraction 0.0 : Accuracy 10%  



