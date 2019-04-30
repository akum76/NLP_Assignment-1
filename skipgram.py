#%%
# -*- coding: utf-8 -*-
from __future__ import division
import random
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
import spacy
import re
import math
import pickle

__authors__ = ['Abhishek Singh','Ahmad El Chaar','Rebecca Erbanni']
__emails__  = ['b00748269@essec.edu','b00739600@essec.edu','b00746038@essec.edu']


nlp = spacy.load("en")

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    original_sentences = []
    processed_sentence=[]
    processed_sentences=[]

    with open(path,encoding='utf-8') as file:
        for line in file:
            #removes extra space before a quote, i.e. makes doesn 't >>> doesn't
            temp = re.sub('\s\'','',line)
            temp = temp.lower()
            #can be deleted later, but kept for the sake of debugging
            original_sentences.append(temp)
    for sentence in original_sentences:
            sentence = nlp(sentence)
            processed_sentence=[]
            for word in sentence:
                #removes full-word punctuation, numbers and returns lemma
                if word.text!='\n'\
                        and not word.is_punct \
                        and not word.like_num \
                        and word.lemma_ !='-PRON-'\
                        and len(word.text)>1:
                            if len(word.text)<4:
                                #since small words don't have a significant lemma like u.s.
                                processed_sentence.append(word.text)
                            else:
                                processed_sentence.append(word.lemma_)
            processed_sentences.append(processed_sentence)

    return processed_sentences

    
def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class LangModel:

    def __init__(self,n):
        """ n is size of context"""
        self.n = n
        self.count = {}

    def train(self,sequences):
        """ max likelihood estimation """
        for sequence in sequences:
            seq = sequence 
            for i in range(0,len(seq)):
                if i in range(0,self.n):
                    left_w=seq[0:i]
                    right_w=seq[i+1:i+self.n+1]
                    ctxt=left_w+right_w
                    ctxt=tuple(ctxt)
                else:
                    if i in range(len(seq)-self.n,len(seq)):
                        left_w=seq[i-self.n:i]
                        right_w=seq[i+1:len(seq)]
                        ctxt=left_w+right_w
                        ctxt=tuple(ctxt)
                    else:
                        left_w=seq[i-self.n:i]
                        right_w=seq[i+1:i+self.n+1]
                        ctxt=left_w+right_w
                        ctxt=tuple(ctxt)
                if ctxt not in self.count:
                    self.count[ctxt] = defaultdict(int)
                self.count[ctxt] [ seq[i]] += 1
        ctxts = self.count.keys()
        for ctxt in ctxts:
            options = self.count[ctxt]
            norm = float(sum(options.values()))
            self.count[ctxt] = dict((k,v/norm) for k,v in options.items())

            
class UnigramTable:
    def __init__(self, wordHash,vocab):
        sumOfWeights = sum([math.pow(w, 3/4) for w in vocab.values()])
        tableSize = int(1e6)
        table = np.zeros(tableSize, dtype=np.uint32)
        k=0
        for j,word in enumerate(wordHash):
            w_count=vocab.get(word,'NA')
            p = 0
            i = 0
            p = float(math.pow(w_count, 3/4))/sumOfWeights
#            print(p)
            for i in range(0,int(p*tableSize)):
#                print(j)
                table[k] = j
                i += 1
                k +=1
        self.table = table[0:k]
        
    def negativeSample(self, count, wordHash):
        count=count
        word=wordHash
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        a = [self.table[i] for i in indices]
        for i in range(0,len(a)):
            if a[i]==word:
                a=self.negativeSample(count,word)
        return a 

        

class SkipGram:
    def __init__(self,sentences, nEmbed=100):
        self.sentences=sentences
        self.nEmbed=nEmbed
        self.vocab_dictionnary={}
        self.wordCount=list()
        self.wordList=list()
        self.wordHash=dict()
        self.totalwordcount=int()
        self.name_cont_neg=[[],[],[]]
        self.hashed_cont_neg=[[],[],[]]
        self.Hashword=dict()
        self.test_data=list()
        self.Loss=list()
        self.MSE=list()
        
    
    def get_vocab_dictionnary(self):
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.vocab_dictionnary:
                    self.vocab_dictionnary.update({word: 1}) #or n_times[word] = 1
                else:
                    self.vocab_dictionnary[word]+= 1
        self.vocab_dictionnary = dict((k, v) for k, v in self.vocab_dictionnary.items())
        self.totalwordcount = len(self.vocab_dictionnary)


    def sigmoid(self,x):
        return np.divide(1,np.add(np.exp(-x),1))

    def train(self,stepsize, epochs):
                
        for x in range (epochs):

            r = list(range(len(self.hashed_cont_neg[0])))

            random.shuffle(r)
            
            for i in r:
                
               word=self.hashed_cont_neg[0][i]
               context=self.hashed_cont_neg[1][i]
               negative=self.hashed_cont_neg[2][i]
               
               word_embed = np.take(self.W,word,axis=1)  #shape 1 by 100
               
               negative_context= [val for sublist in [negative,context] for val in sublist]
                             
               negative_context_embed= np.take(self.W_prime,negative_context,axis=1)

               activation = self.sigmoid(np.dot(negative_context_embed.T,word_embed))

               zeros_ones = [np.zeros(len(negative),dtype=int).tolist(),np.ones(len(context),dtype=int).tolist()]

               zeros_ones= [val for sublist in zeros_ones for val in sublist]

               dl_dc= np.multiply(np.add(zeros_ones,np.multiply(activation,-1)),stepsize)

               dl_dw=np.sum(np.multiply(negative_context_embed,dl_dc),axis=1,keepdims=True)

               temp_dl_dc = np.multiply(np.ones([len(self.W.T[0]),len(dl_dc)]),dl_dc)
               
               self.W_prime.T[negative_context]+=np.multiply(word_embed,temp_dl_dc.T)
                                                                        
               self.W.T[[word]]=+ dl_dw.T
                                  
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def initialize_embedding(self,):
        self.W = np.random.rand(self.nEmbed,len(self.vocab_dictionnary))
        self.W_prime = np.random.rand(self.nEmbed,len(self.vocab_dictionnary))
    
        
    #defining a function here that takes care of hashing and counts the frequency of words
    def hashing(self, sentences):
        i=0
        for sentence in sentences:
            for word in sentence:
                if word not in self.wordList:
                    self.wordHash[word] = i  # The length of the list is used as our hash/counter as well
                    self.wordCount.append(1)     # We append the Word count
                    self.wordList.append(word)     # We append the word
                    i=i+1
                else:
                    self.wordCount[self.wordList.index(word)] += 1 
    

    def make_hashed_list(self,ngram,neg_model,n=2):
        for Keys, Values in zip(ngram.keys(),ngram.values()):
            for Value in  Values:        
                self.number_of_negative_samples = len(Keys)*n
                self.hashed_cont_neg[0].append(self.wordHash.get(Value,-1))
                temp=[]
                for Key in Keys:
                    temp.append(self.wordHash.get(Key,-1))
                self.hashed_cont_neg[1].append(temp)
                self.hashed_cont_neg[2].append(neg_model.negativeSample(self.number_of_negative_samples,self.wordHash.get(Value)))
    
    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        try:
            word1=nlp(word1)[0].lemma_
            word1=self.wordHash[word1]
        except:
            word1=np.random.randint(0,len(self.wordHash))
        
        try:
            word2=nlp(word2)[0].lemma_
            word2=self.wordHash[word2]
        except:
            word2=np.random.randint(0,len(self.wordHash))
        
        word1=self.W.T[[word1]]
        word2=self.W.T[[word2]]
        
        return np.divide(np.dot(word1,word2.T),np.multiply(np.linalg.norm(word1),np.linalg.norm(word2)))[0][0]


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            a = pickle.load(f)  
        return a


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        tokenized_sentences = text2sentences(opts.text)
        sg = SkipGram(tokenized_sentences,nEmbed=100)
        sg.hashing(tokenized_sentences)
        sg.get_vocab_dictionnary()
        neg_model=UnigramTable(sg.wordHash,sg.vocab_dictionnary)
        ngram = LangModel(3)
        ngram.train(tokenized_sentences)
        ngram=ngram.count
        sg.make_hashed_list(ngram,neg_model,4)
        sg.initialize_embedding()
        sg.train(0.02,5)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
                print (sg.similarity(a,b))