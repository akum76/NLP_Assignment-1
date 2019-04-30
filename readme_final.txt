########################################################################################################
########################################################################################################
########################################################################################################
Project Pipeline:
- text2sentences <- Makes tokenization of sentences
sample code for 1000 sentences:
[original_sentences, tokenized_sentences] = text2sentences(path,1000)

- skipgram <- Incorporates tokenized sentences in skipgram class
creates the model
model = SkipGram(tokenized_sentences)

- creates the hashing for each token (1 token <> 1 hash)
model.hashing(tokenized_sentences)

- makes the vocab list:
model.get_vocab_dictionnary()

- launches unigram class and creates the table that includes probability 3/4 for negative samples
neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)

- Makes the ngram table (context words)
in the example below, we used a bigram model <> 2
ngram = LangModel(2)
ngram.train(tokenized_sentences)

- creates the word/context/negative sample table to be used later on embeddings
the example below, we used 3 negative samples for each positive sample
model.make_hashed_list(ngram,neg_model,3)

- creates the embedding matricies
model.initialize_embedding()

- Trains the model and updates word and context embeddings
learning rate is 0.01, 1 epoch run
model.train(0.01,1)

#Optional for testing the data, the below will
give back the table with similarity compared to the ones obtained from 
the EN-SIMLEX-999 database

#Testing the data
for x in range(5):
    print("epochs: ", x)
    model.test()
    model.train(0.01,1)
    model.MSE_Calculate()

print(model.test_data)
print(model.MSE)

#############################################
text2sentences (function):
Takes in - path
	 - number of sentences to process

Returns tokenized sentences
Tokenization consist of - remove punctuation words
			- removing words that are numbers
			- getting the lemma of the word

############################################
LangModel (class):
Takes in n for ngram count (bigram,trigram...)

function train
Takes in - tokenized sentences (obtained from text2sentences)

returns dictionnary
dictionnary consist of context and words

############################################
UnigramTable (class):

1) 
Initialization: 
Initialize the variable sumOfWeights which act as the denominator in the formula to calculate probability
p(w) of a word getting picked from the vocabulary

Create a zero vector of size 1e6 to store the words from which we will randomly pick negative samples

Fill the large vector with word hashes by filling each word p(w)*1e6 times so probability pf picking the 
word remains same.

2)
negativeSample:

This function takes in the count of negative samples required and the word for which we need negative samples

The function selects negative samples from the large vector created earlier and avoids picking the target word
by using recursive call of the function if the word gets picked  

############################################
Skipgram (class):
1)
function get_vocab_dictionnary
Takes in sentences
returns dictionnary with all vocab words and the count for each word
2)
function load_test_data
Takes in ARGS of class
returns list (called test_data), which contains a list of pair of words and the score as per the EN-SIMLEX-999
3)
function test
Takes in ARGS of class
returns appends to test_data the score computed using the similarity function of the model
4)
function MSE_Calculate
Takes in ARGS of class
returns Mean square error on the latest scores computed by similarity against those obtained from the original 
EN-SIMLEX-999 file
5)
function sigmoid
Takes in ARGS of class, a numpy array
returns sigmoid calculation of the array
note that the function was written in a way, that if provided with a numpy array containing several lists of numbers
it will return the sigmoid of each individual list
6)
function train
Takes in ARGS of class, stepsize, epochs
updates word and context embedding using the w2v gradient descent as follows:
 for loop on epochs
  for loop on number of samples
   a) captures a random sample from the list hashed_cont_neg (contains hashed word-contexts-negative samples)
   b) captures the embeddings of word, context, negative sample 
   c) applies sigmoid on context and negative sample
   d) makes the vector zeros-ones which contains zeros in the lengh of negative samples and 
      ones in the length of context embeddings (Labels)
   e) computes the derivative dl_dc (in respect of context and negative samples) computed as (label - sigmoid) * learning rate
   f) computes the derivative of word embedding dl_dw = dl_dc * context and negative sample embedding
   g) updates the context and negative sample embedding as dl_dc * word embedding
   h) updates word embedding = dl_dw
7)
function save
Take in path
Saves the class under path
8)
function initialize embedding
Takes in ARGS of class
creates the word and context embeddings using vocab length and embedding length defined earlier in the class
9)
function random_word
Takes in word hash
returns a random word from the word vocab setting all vocab words with the same probability of occurence
10)
function hashing
Takes in tokenized sentences
returns dictionnary wordHash containing a word and its corresponding hash
11)
function inverse_hashing
Takes in wordHash
returns dictionnary Hashword containing a hash and its corresponding word
12)
function make_hashed_list

This function is used to create a list which has all the words of the vocabulary, positive context and the negative samples.

This utilises the Unigram table mentioned earlier for negative contexts.

returns a list hashed_cont_neg which has 3 columns of word, positive and negative samples

13)
function similarity
Takes in a pair of words
returns the cosine similarity using the embeddings
14)
function inverse_similarity
Takes in a pair of hashes
returns the cosine similarity using the embeddings
15)
function load
Take in path
load the class saved under path

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
The below section is the code we used in our experiments:
Note that two measures were used, the loss function and the MSE against the 100 words obtained from
EN-SIMLEX-999.txt dataset
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

#########################################################################################################################
#CODE 1: Check the best alpha to use
#########################################################################################################################
#%%
#Checking which alpha is best
alphas=np.logspace(0.2,0.25,2)-1

for y in alphas:
    model.Loss=list()
    model.MSE=list()
    model.initialize_embedding()
    model.load_test_data()
    for x in range(10):
        print("epochs: ", x)
        model.train(y,1)
        model.test()
        model.MSE_Calculate()
    df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
    df.to_csv(str(round(y,5))+".csv", sep=',',index=False)
    model.save('alpha'+str(round(y,5)))

#########################################################################################################################
#CODE 2: Check the evolution across epochs
#########################################################################################################################
#%%
#Checking which epoch is best
model.initialize_embedding()
model.load_test_data()
for x in range(40):
    print("epochs: ", x)
    model.train(0.1,1)
    model.test()
    model.MSE_Calculate()
df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
df.to_csv("Epochs Alpha at 0.1.csv", sep=',',index=False)

########################################################################################################
#CODE 3: Check the evolution across number of sentences
########################################################################################################
#%%
#Check Number of Sentences
sentcount=[5000,10000,20000,30000]    
for x in sentcount:
    #gets the tokized sentences
    [original_sentences, tokenized_sentences] = text2sentences(path,x)
         
    #loads sentences to skip gram
    model = SkipGram(tokenized_sentences)
    
    #hash sentences
    model.hashing(tokenized_sentences)
    model.inverse_hashing()
    
    #makes vocab list
    model.get_vocab_dictionnary()
    
    #Populate Negative Samples Table
    neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)
    
    #makes ngram
    ngram = LangModel(2)
    ngram.train(tokenized_sentences)
    ngram=ngram.count
    
    #make list for neural network (word,context,negative samples), takes in number
    #of negative samples to give for each context sample
    model.make_hashed_list(ngram,neg_model,3)
    
    #initialize neural network
    model.initialize_embedding()
    model.load_test_data()

    #Populate test dataset
    for y in range(5):
        print("epochs: ", y)
        model.train(0.1,1)
        model.test()
        model.MSE_Calculate()
    df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
    df.to_csv("sentence count"+str(x)+".csv", sep=',',index=False)

#########################################################################################################################
#CODE 4: Check the evolution across number of embeddings
#########################################################################################################################
#%%
#Number of Embeddings
#gets the tokized sentences
[original_sentences, tokenized_sentences] = text2sentences(path,30000)
#loops on embedding
embedcount=[100,200,300]    
for x in embedcount:
         
    #loads sentences to skip gram
    model = SkipGram(tokenized_sentences,nEmbed=x)
    
    #hash sentences
    model.hashing(tokenized_sentences)
    model.inverse_hashing()
    
    #makes vocab list
    model.get_vocab_dictionnary()
    
    #Populate Negative Samples Table
    neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)
    
    #makes ngram
    ngram = LangModel(2)
    ngram.train(tokenized_sentences)
    ngram=ngram.count
    
    #make list for neural network (word,context,negative samples), takes in number
    #of negative samples to give for each context sample
    model.make_hashed_list(ngram,neg_model,3)
    
    #initialize neural network
    model.initialize_embedding()
    model.load_test_data()

    #Populate test dataset
    for y in range(5):
        print("epochs: ", y)
        model.train(0.1,1)
        model.test()
        model.MSE_Calculate()
    df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
    df.to_csv("sentence count"+str(x)+".csv", sep=',',index=False)

#########################################################################################################################
#CODE 5: Check the evolution across different ngrams
#########################################################################################################################
#%%
#Number of Ngram
for x in range(2,5):
         
    #loads sentences to skip gram
    model = SkipGram(tokenized_sentences)
    
    #hash sentences
    model.hashing(tokenized_sentences)
    model.inverse_hashing()
    
    #makes vocab list
    model.get_vocab_dictionnary()
    
    #Populate Negative Samples Table
    neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)
    
    #makes ngram
    ngram = LangModel(x)
    ngram.train(tokenized_sentences)
    ngram=ngram.count
    
    #make list for neural network (word,context,negative samples), takes in number
    #of negative samples to give for each context sample
    model.make_hashed_list(ngram,neg_model,3)
    
    #initialize neural network
    model.initialize_embedding()
    model.load_test_data()

    #Populate test dataset
    for y in range(5):
        print("epochs: ", y)
        model.train(0.02,1)
        model.test()
        model.MSE_Calculate()
    df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
    df.to_csv("NGram "+str(x)+".csv", sep=',',index=False)
#########################################################################################################################
#CODE 6: Check the evolution across multiple of negative context against each positive context word
#########################################################################################################################
#%%
#Number of Negative contexts
for x in range(4,10,2):         
    #loads sentences to skip gram
    model = SkipGram(tokenized_sentences)
    
    #hash sentences
    model.hashing(tokenized_sentences)
    model.inverse_hashing()
    
    #makes vocab list
    model.get_vocab_dictionnary()
    
    #Populate Negative Samples Table
    neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)
    
    #makes ngram
    ngram = LangModel(3)
    ngram.train(tokenized_sentences)
    ngram=ngram.count
    
    #make list for neural network (word,context,negative samples), takes in number
    #of negative samples to give for each context sample
    model.make_hashed_list(ngram,neg_model,x)
    
    #initialize neural network
    model.initialize_embedding()
    model.load_test_data()

    #Populate test dataset
    for y in range(5):
        print("epochs: ", y)
        model.train(0.02,1)
        model.test()
        model.MSE_Calculate()
    df = pandas.DataFrame(data={"Loss": model.Loss, "MSE": model.MSE,  "Epochs": range(len(model.MSE))})
    df.to_csv("neg contexts "+str(x)+".csv", sep=',',index=False)


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

We removed the loss function Calculation, MSE, random_word and few other parts that are not needed 
from the code submitted in order to send a clean version for submission, 
Please find below the original code with all the other functionalities
and marked with #### the parts removed


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
import pandas


__authors__ = ['Abhishek Singh','Ahmad El Chaar','Rebecca Erbanni']
__emails__  = ['b00748269@essec.edu','b00739600@essec.edu','b00746038@essec.edu']


nlp = spacy.load("en")

def text2sentences(path,number_of_sentences=1000):
    # feel free to make a better tokenization/pre-processing
    x=0
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
            x=x+1
            if x >number_of_sentences:
                break
    for sentence in original_sentences:
            sentence = nlp(sentence)
            processed_sentence=[]
            for word in sentence:
                #removes stop words, full-word punctuation, numbers and returns lemma
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

    return [original_sentences, processed_sentences]

    
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
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 0):
        self.sentences=sentences
        self.nEmbed=nEmbed
        self.negativeRate=negativeRate
        self.winSize=winSize
        self.minCount=minCount
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
        self.vocab_dictionnary = dict((k, v) for k, v in self.vocab_dictionnary.items() if v >= self.minCount)
        self.totalwordcount = len(self.vocab_dictionnary)
####################################################################################################################
    def load_test_data(self):
        #load testing data
        test_data=list()
        with open('EN-SIMLEX-999.txt') as inputfile:
            for line in inputfile:
                test_data.append(line.strip().split('\t'))
        
        for x in range (len(test_data)):
            test_data[x][2]=float(test_data[x][2])/10
        
        self.test_data = test_data

    def test(self):
        for x in range (len(self.test_data)):
            self.test_data[x].append(self.similarity(self.test_data[x][0],self.test_data[x][1]))
########################################################################################################
    def MSE_Calculate(self):
        mse=0
        for x in range(len(self.test_data)):
            mse=+(float(self.test_data[x][2])-float(self.test_data[x][len(self.test_data[0])-1]))**2
        self.MSE.append(mse/len(self.test_data))                
####################################################################################################################

    def sigmoid(self,x):
        return np.divide(1,np.add(np.exp(-x),1))

    def train(self,stepsize, epochs):
                
        for x in range (epochs):

            r = list(range(len(self.hashed_cont_neg[0])))
            
            negative_counter=0
            positive_counter=0
            positive_loss=0
            negative_loss=0
            
            random.shuffle(r)
            
            for i in r:
                
               word=self.hashed_cont_neg[0][i]
               context=self.hashed_cont_neg[1][i]
               negative=self.hashed_cont_neg[2][i]
               
               word_embed = np.take(self.W,word,axis=1)  #shape 1 by 100
               
               negative_context= [val for sublist in [negative,context] for val in sublist]
              

               negative_embed = np.take(self.W_prime,negative,axis=1)
               context_embed =  np.take(self.W_prime,context,axis=1)

               
               negative_context_embed= np.take(self.W_prime,negative_context,axis=1)

               activation = self.sigmoid(np.dot(negative_context_embed.T,word_embed))

               zeros_ones = [np.zeros(len(negative),dtype=int).tolist(),np.ones(len(context),dtype=int).tolist()]

               zeros_ones= [val for sublist in zeros_ones for val in sublist]

               dl_dc= np.multiply(np.add(zeros_ones,np.multiply(activation,-1)),stepsize)

               dl_dw=np.sum(np.multiply(negative_context_embed,dl_dc),axis=1,keepdims=True)

               temp_dl_dc = np.multiply(np.ones([len(self.W.T[0]),len(dl_dc)]),dl_dc)
               
               self.W_prime.T[negative_context]+=np.multiply(word_embed,temp_dl_dc.T)
                                                                        
               self.W.T[[word]]=+ dl_dw.T
               
               positive_loss=+np.sum(np.log(self.sigmoid(np.dot(word_embed.T,context_embed))))
               negative_loss=+np.sum(np.log(self.sigmoid(np.dot(-word_embed.T,negative_embed))))
               positive_counter=+ context_embed.shape[1]
               negative_counter=+ negative_embed.shape[1]
                             
            self.Loss.append(-np.add(np.divide(positive_loss,positive_counter),np.divide(negative_loss,negative_counter)))
               
                                  
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def initialize_embedding(self,):
        self.W = np.random.rand(self.nEmbed,len(self.vocab_dictionnary))
        self.W_prime = np.random.rand(self.nEmbed,len(self.vocab_dictionnary))
    
########################################################################################################
    def random_word(self,word_hash):
        random_hash = np.random.randint(0,self.totalwordcount) 
        return random_hash if random_hash != word_hash else self.random_word(word_hash)
########################################################################################################        
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
                    self.wordCount[self.wordList.index(word)] += 1 # and keep track of frequency for mincount filtering later                   
    
    def inverse_hashing(self):
        self.Hashword= {v: k for k, v in self.wordHash.items()}

    def make_hashed_list(self,ngram,neg_model,n=2):
        for Keys, Values in zip(ngram.keys(),ngram.values()):
            for Value in  Values:        
                self.number_of_negative_samples = len(Keys)*n
                self.hashed_cont_neg[0].append(model.wordHash.get(Value,-1))
                temp=[]
                for Key in Keys:
                    temp.append(model.wordHash.get(Key,-1))
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
            word1=self.wordHash[word1]
        except:
            word1=np.random.randint(0,len(self.wordHash))
        
        try:
            word2=self.wordHash[word2]
        except:
            word2=np.random.randint(0,len(self.wordHash))

        
        word1=self.W.T[[word1]]
        word2=self.W.T[[word2]]
        
        return np.divide(np.dot(word1,word2.T),np.multiply(np.linalg.norm(word1),np.linalg.norm(word2)))[0][0]

########################################################################################################
    def inverse_similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """        
        word1=self.W.T[[word1]]
        word2=self.W.T[[word2]]
        
        return np.divide(np.dot(word1,word2.T),np.multiply(np.linalg.norm(word1),np.linalg.norm(word2)))
########################################################################################################
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            a = pickle.load(f)  
        return a

#%%        
#update director
import os
os.chdir("C:/Users/Lenovo_student/Documents/One_Drive/OneDrive - American University of Beirut/Education/ESSEC-Central/Natural Language Processing/Assignments/Assignment 1")

#load 1 file from the billion words tokens here
path=('news.en-00001-of-00100')

#gets the tokized sentences
[original_sentences, tokenized_sentences] = text2sentences(path,10)
     
#loads sentences to skip gram
model = SkipGram(tokenized_sentences)

#hash sentences
model.hashing(tokenized_sentences)
model.inverse_hashing()

#makes vocab list
model.get_vocab_dictionnary()

#Populate Negative Samples Table
neg_model=UnigramTable(model.wordHash,model.vocab_dictionnary)

#%%
#makes ngram
ngram = LangModel(2)
ngram.train(tokenized_sentences)
ngram=ngram.count

#make list for neural network (word,context,negative samples), takes in number
#of negative samples to give for each context sample
model.make_hashed_list(ngram,neg_model,3)

#initialize neural network
model.initialize_embedding()

#Populate test dataset
model.load_test_data()



#%%
#Testing the data
for x in range(5):
    print("epochs: ", x)
    model.train(0.01,1)
    model.test()
    model.MSE_Calculate()


########################################################################################################
########################################################################################################
########################################################################################################

The below code takes the skipgram model, the word (string) and a number n and returns the n closest word
in terms of cosine similarity

########################################################################################################
########################################################################################################
########################################################################################################


def closest_words(model,w,K=5):
        # K most similar words: self.score  -  np.argsort
    
        #first stores distances  each being compared and the distance in arrays
        distances=np.array([])
        words=np.array([])
        w=model.wordHash[w]
        w=model.W.T[[w]]
        
        for i in model.vocab_dictionnary.keys():
            v2=model.wordHash[i]
            v2=model.W.T[[v2]]
            returned_score = np.dot(w,v2.T)/(linalg.norm(w)*linalg.norm(v2))
            distances = np.append(distances,returned_score)
            words=np.append(words,i)

        #Finds the lowest distance with argpartition
        #then converts that into a word, using the array
        
        closest_K_words = words[distances.argsort()[-K:][::-1]]
        return closest_K_words



#############################################
#############################################
#############################################

References:

https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Description:This helped us Understand how the loss function is desinged and the function of the derivatives we need to compute

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Description: This article explains how to design the skipgram. We used snippets of parts to implement the derivative function tweaking it for our case

Explained: Deriving Mikolov et al.’s
Negative-Sampling Word-Embedding Method
Yoav Goldberg and Omer Levy
Description: This helped us understanding the concepts and negative sampling and how to implement it in Skipgram

https://www.youtube.com/watch?v=TaZz_K2xJy8

Desription: This video by Andrew Ng also explains the implementation of skipgram with negative sampling and a practical way to implement

#############################################
#############################################
#############################################
