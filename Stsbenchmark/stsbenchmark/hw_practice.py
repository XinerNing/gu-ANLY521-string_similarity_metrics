# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:22:58 2019

@author: shera
"""
import nltk
from nltk.metrics.distance import edit_distance
#from nltk.metrics.distance import binary_distance
from nltk.translate import nist_score, bleu_score
from scipy.stats.stats import pearsonr   
import argparse
texts = []
labels = []

with open('sts-dev.csv', 'r',encoding="utf-8") as dd:
    for line in dd:
        fields = line.strip().split("\t") #.strip():remove white space before and after the str
        labels.append(float(fields[4]))
        t1 = fields[5].lower()
        t2 = fields[6].lower()
        texts.append((t1, t2))

## function to get nist score
def nist_func(x,y):
    try:
        return nist_score.sentence_nist([nltk.word_tokenize(x)],nltk.word_tokenize(y))
    except ZeroDivisionError:
        return 0



lev_dist=[]
wer_score=[]
mynist_score=[]
mybleu_score=[]
for pair in texts:
    t1,t2=pair
    token1=nltk.word_tokenize(t1)
    token2=nltk.word_tokenize(t2)
    dist=edit_distance(t1,t2)
    dist_new=edit_distance(token1,token2)
    #t1=t1.split()
    #t2=t2.split()
    #mywer=wer(t1.split(),t2.split())
    mynist=nist_func(t1,t2)
    mybleu=bleu_score.sentence_bleu([token1],token2)
    mywer=((dist_new)/len(token1))+((dist_new)/len(token2))
    lev_dist.append(dist)
    wer_score.append(mywer)
    mynist_score.append(mynist)
    mybleu_score.append(mybleu)

print(pearsonr(labels,lev_dist)) # gives corr and p-value
#print(pearsonr(labels,wer_score))
print(pearsonr(labels,mynist_score))
print(pearsonr(labels,mybleu_score))
print(pearsonr(labels,wer_score))

