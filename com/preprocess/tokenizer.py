'''
Created on Oct 22, 2017

@author: krish.mahajan
'''


import re 




def tokenizer(s,stopwords):  
    """
    This function 'll clean each individual text and convert in 
    to tokens
    """
    s = re.sub("[^a-zA-Z]"," ",s)
    s = s.lower() 
    tokens=s.split(' ') 
    tokens = [t for t in tokens if len(t)>2] 
    tokens = [token for token in tokens if token not in stopwords] 
    return  tokens  

# def tokenize_data(train_data,test_data,min_freq):
def tokenize_test_data(data,word_index_map,min_freq,stopwords):
    
    tokenized=[[]]    
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text'],stopwords) 
        for token in tokens[:]: 
            if token not in word_index_map:
                tokens.remove(token)
        tokenized.append(tokens) 
        
    return tokenized

def tokenize_train_data(data,min_freq,stopwords):
    tokenized = [[]]
    word_index_map = {} 
    current_index = 0 

    '''Reading all the documents(train + test) and tokenizing them and adding 
    to dictionary
    + Also making grand vocabularly V
    ''' 
            
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text'],stopwords) 
        data.loc[index,'text']=" ".join(tokens)
        for token in tokens: 
            if token not in word_index_map:
                word_index_map[token] =[current_index ,1]
                current_index +=1 
            word_index_map[token][1] += 1  

        
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text'],stopwords) 
        label = row['label']
        tokens.append(label)
        for token in tokens[:-1]: 
            if word_index_map[token][1]<=min_freq:
                tokens.remove(token)
        tokenized.append(tokens)  
                
    word_index_map_new={} 
    i=0
    for key in word_index_map.keys():
        if word_index_map[key][1]>min_freq:
            word_index_map_new[key]=i 
            i+=1
    word_index_map=word_index_map_new
                

    return data,word_index_map,tokenized

