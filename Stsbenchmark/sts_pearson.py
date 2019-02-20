# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:11:54 2019

@author: shera
"""
import nltk
from nltk.metrics.distance import edit_distance
from nltk.translate import nist_score, bleu_score
from scipy.stats.stats import pearsonr  
from difflib import SequenceMatcher  #longest common substring
import argparse

## function to get nist score
def nist_func(x,y):
    try:
        return nist_score.sentence_nist([nltk.word_tokenize(x)],nltk.word_tokenize(y))
    except ZeroDivisionError:
        return 0

## function to get the longest common substring
def find_LCS(x,y):
    match = SequenceMatcher(None, x,y).find_longest_match(0, len(x), 0, len(y))
    common_str=x[match.a: match.a + match.size]
    return (len(common_str))


def main(sts_data, output_file):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""
    texts = []
    labels = []

    with open(sts_data, 'r',encoding="utf-8") as dd:
        for line in dd:
            fields = line.strip().split("\t") #.strip():remove white space before and after the str
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))
    lev_dist=[]
    wer_score=[]
    mynist_score=[]
    mybleu_score=[]
    myLCS_score=[]
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
        myLCS = find_LCS(t1,t2)
        lev_dist.append(dist)
        wer_score.append(mywer)
        mynist_score.append(mynist)
        mybleu_score.append(mybleu)
        myLCS_score.append(myLCS)

    Lev_corr=round(pearsonr(labels,lev_dist)[0],3) # gives corr and p-value
    NIST_corr=round(pearsonr(labels,mynist_score)[0],3)
    BLEU_corr=round(pearsonr(labels,mybleu_score)[0],3)
    WER_corr=round(pearsonr(labels,wer_score)[0],3)
    LCS_corr=round(pearsonr(labels,myLCS_score)[0],3)
    mycorr_list=[NIST_corr,BLEU_corr,WER_corr,LCS_corr,Lev_corr]

    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]
    


    with open(output_file, 'w') as out:
        out.write(f"Semantic textual similarity for {sts_data}\n")
        for i in range(0,len(score_types)):
            out.write(f"{score_types[i]} correlation: {mycorr_list[i]}\n")
        
        # TODO: write scores. See example output for formatting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-test.csv",
                        help="tab separated sts data in benchmark format")
    parser.add_argument("--output_file", type=str, default="test_output.txt",
                        help="report on string similarity ")
    args = parser.parse_args()

    main(args.sts_data, args.output_file)