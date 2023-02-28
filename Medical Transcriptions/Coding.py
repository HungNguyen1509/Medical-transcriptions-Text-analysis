# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:17:44 2022

@author: nqhun
"""

import pandas as pd
import numpy as np
import os
import nltk as nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import ConditionalFreqDist,FreqDist
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('omw-1.4')
path = 'C:/Users/nqhun/Downloads/mtsamples/data'
os.chdir(path)
find = ['body', 'father', 'brother', 'sister', 'aunt', 'grandfather', 'grandmother', 'uncle', 'son', 
	       'daughter', 'cousin', 'mom', 'dad', 'nephew', 'niece']
list_stop_words = list(stopwords.words('english'))
list_words_stop = ['a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','the','to','was','were','will','with']
class NeuralLangue():
    #path_file is path 
    def __init__(self,path_file):
        self.path_file = path_file
        self.df = self.read_text_to_dataframe().copy()
        
    def read_text_to_dataframe(self):
        path_file=self.path_file
        os.chdir(self.path_file)
        os.chdir(path_file)
        df = pd.DataFrame(columns = ['Namefile','Text'])
        #index names are listnames of textfiles
        df['Namefile'] = os.listdir()
        df.index= os.listdir()
        i = 0
        file_path = self.path_file
        for file in os.listdir():
            with open(file_path+'/'+file,'r') as f:
                if file.endswith(".txt"):
                    df['Text'][i] = str(f.read())
                    i = i+1
        return df

    def read_text_file(self,name_file):
        df = pd.DataFrame(index =[name_file], columns = ['Namefile','Text'])
        path_file = self.path_file
        os.chdir(path_file)
        with open(path_file+'/'+name_file,'r') as f:
            text = str(f.read())
        df.iloc[0,1]=text
        df.iloc[0,0]=name_file
        return df
  
    def paragraph_to_pharase_dataframe(self,name_file = False):
        if name_file:
            result = self.read_text_file(name_file)
        else:
            result =self.df
        return result

    def paragraph_to_phrase_dictionary(self,name_file = False):
        text = self.paragraph_to_pharase_dataframe(name_file)
        dict_text = {}
        for i in range(np.shape(text)[0]):
            word = [word_tokenize(t) for t in sent_tokenize(text.iloc[i,1])]
            dict_text[i] = word
        result = dict_text
        return result

    def frequence_of_list_words(self,list_words_want_to_find,name_file = False):
        data = pd.DataFrame(index = self.df.index,columns=list_words_want_to_find)
        df = self.df
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for i in range(np.shape(df)[0]):
            print(i)
            word_with_lowcase= df.iloc[i,1].lower()
            tokens = tokenizer.tokenize(word_with_lowcase)
            new_list = [x for x in tokens if (x not in list_words_stop)]
            word_with_lowcase_and_turn_to_singular = [lemmatizer.lemmatize(t) for t in new_list]
            frequence = FreqDist(word_with_lowcase_and_turn_to_singular)
            value = [frequence[word] for word in list_words_want_to_find]
            data.iloc[i,:] = value
        return data