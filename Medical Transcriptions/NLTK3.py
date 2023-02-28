# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:48:12 2022

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
from nltk.tokenize import MWETokenizer
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
from nltk.corpus import names
nltk.download()

male = names.words('male.txt')
female = names.words('female.txt')
nltk.download('omw-1.4')
path = 'C:/Users/nqhun/Downloads/mtsamples/data'
os.chdir(path)
find = [' father ', ' brother ', ' sister ', ' aunt ', ' grandfather ', ' grandmother ', ' uncle ', ' son ', 
	       ' daughter ', ' cousin ', ' mom ', ' dad ', ' nephew ', ' niece ', 
           ' brothers ', ' sisters ', ' aunts ', ' grandfathers ', ' grandmothers ', ' uncles ', ' sons ', 
	       ' daughters ', ' cousins ', ' nephews ', ' nieces ']
list_stop_words = list(stopwords.words('english'))
list_words_stop = ['a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','the','to','was','were','will','with']
list_diseases = ['breast cancer', 'ADHD', 'HTN',
                 'Breast cancer', 'bipolar', 'hypertension',
                 'CA', 'bipolar disorder', 'brain aneurysm',
                 'cancer', 'depressed',	'cerebral aneurysm',
                 'colon cancer', 'depression', 'cerebrovascular', 'accident',
                 'gastric carcinoma', 'mental illness', 'stroke',
                 'lung cancer', 'mood disorder', 'strokes'
                 'prostate cancer',	'mood disorder/bipolar', 'adult-onset diabetes',
                 'renal CA', 'nervous breakdowns', 'diabetes',
                 'throat cancer', 'Schizophrenia', 'diabetes mellitus',
                 'CHF', 'suicide', 'DM',
                 'CAD',	'coronary heart disease', 'type 2 diabetes',
                 'acute myocardial infarction', 'heart attack', 'alcohol abuse',
                 'congestive heart failure', 'heart disease', 'alcoholic',
                 'coronary artery disease', 'Heart disease', 'alcoholism',
                 'myocardial infarction', 'heart failure',	'alcohol to excess',
                 'valvular heart disease', 'MI', 'alcohol use',
                 'vascular strokes', 'drug addict',	'deceased from alcohol'
                 'substance abuse', 'using substance']
#%%
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
    
    def data_lowcase(self):
        df = self.df
        for i in range(np.shape(df)[0]):
            df.iloc[i,1]=df.iloc[i,1].lower()
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
            word = [word_tokenize(t) for t in sent_tokenize(text.iloc[i,1].lower())]
            dict_text[i] = word
        result = dict_text
        return result
    
    def paragraph_to_phrase_dictionary_with_str(self,name_file = False):
        text = self.paragraph_to_pharase_dataframe(name_file)
        dict_text = {}
        for i in range(np.shape(text)[0]):
            sentence = sent_tokenize(text.iloc[i,1].lower())
            dict_text[i] = sentence
        result = dict_text
        return result
    
    def find_something(self,list_word_to_find,list_disease_to_find,k):
        data=self.paragraph_to_phrase_dictionary_with_str()
        df = self.df
        result = {}
        for i in range(np.shape(df)[0]):
            for j in range(len(data[i])):
                value= {'sentence':[],
                        'type_':[],
                        'list_word_': [],
                        'list_disease': []}
                sentence = []
                enum1 = 0
                type_1 = []
                word1x = []
                wordx = []
                for word in list_word_to_find:
                    if word in data[i][j]:
                        for word1 in list_disease_to_find:
                            if word1 in data[i][j]:
                                enum1 += enum1
                                sentence.append(j)
                                wordx.append(word)
                                word1x.append(word1)
                                if data[i][j].index(word)>data[i][j].index(word1):
                                    type_ = 1
                                else:type_ = 0 
                                type_1.append(type_)
                if enum1>=k and word1x != [] and wordx != []:
                    value['sentence'] = sentence
                    value['type_'] = type_1
                    value['list_word_'] = wordx
                    value['list_disease'] = word1x
                    result[df.index[i]] = value
        return result
                        
    
    def frequence_of_list_words(self,list_words_want_to_find,name_file = False):
        data = pd.DataFrame(index = self.df.index,columns=list_words_want_to_find)
        df = self.df
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        lemmatizer = nltk.stem.WordNetLemmatizer()
        dict_word = {}
        list_frequence = {}
        for i in range(np.shape(df)[0]):
            print(i)
            word_with_lowcase= df.iloc[i,1].lower()
            tokens = tokenizer.tokenize(word_with_lowcase)
            new_list = [x for x in tokens if (x not in list_words_stop)]
            word_with_lowcase_and_turn_to_singular = [lemmatizer.lemmatize(t) for t in new_list]
            dict_word[i] = word_with_lowcase_and_turn_to_singular
            frequence = FreqDist(word_with_lowcase_and_turn_to_singular)
            list_frequence[i] = frequence
            value = [frequence[word] for word in list_words_want_to_find]
            data.iloc[i,:] = value
        return data,dict_word,list_frequence
 
    def tranform_to_2grams(self,list_word):
        list_2_word = []
        for i in range(len(list_word)):
            list_2_ = tuple(word_tokenize(list_word[i]))
            list_2_word.append(list_2_)
        return  list_2_word
    
    def find_disease_frequence(self,list_diseases):
        list_2_word = self.tranform_to_2grams(list_diseases)
        data = pd.DataFrame(index = self.df.index,columns = list_diseases)
        tokenizer = MWETokenizer(list_2_word, separator=' ')
        list_frequence = {}
        df = self.data_lowcase()
        for i in range(np.shape(df)[0]):
            tokens = tokenizer.tokenize(df.iloc[i,1].split())
            frequence = FreqDist(tokens)
            list_frequence[i] = frequence
            value = [frequence[word] for word in list_diseases]
            data.iloc[i,:] = value
        return data
        

    def get_human_names(self,name):
        df = self.df
        data = pd.DataFrame(index = range(np.shape(df)[0]),columns=name)
        
        for i in range(np.shape(df)[0]):
            tokens =word_tokenize(df.iloc[i,1])
            frequence = FreqDist(tokens)
            value = [frequence[word] for word in name]
            data.iloc[i,:] = value
        return data
#%%

df=NeuralLangue(path).read_text_to_dataframe().copy()
a[df.index[0]]['list_word_']


