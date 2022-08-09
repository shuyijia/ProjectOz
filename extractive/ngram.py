from collections import Counter
import itertools
import numpy as np
import math
from datasets import load_dataset

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

 
class Ngram:
    def __init__(self):
        self.START = '<START>'
        self.STOP = '<STOP>'
        self.unigram_dict = None
        
    def query_ngrams(self, query, n):
        query_split = query.split()
        if n ==1:
            return query_split
        elif n == 2:
            return [query_split[i:i+2] for i in range(len(query_split)-1)]
        elif n == 3:
            return [query_split[i:i+3] for i in range(len(query_split)-2)]
        
    def compute_ngram(self, sents, n):

        ngram_set = None
        ngram_dict = None
        if not self.unigram_dict:
            ngram_list = [s[i:i+n] for s in sents for i in range(len(s)-n+1)]
            ngram_list = list(itertools.chain(*ngram_list))
            self.unigram_dict = Counter(ngram_list) 
        if n == 1:
            ngram_list = [s[i:i+n] for s in sents for i in range(len(s)-n+1)]
            ngram_list = list(itertools.chain(*ngram_list))
        elif n >= 2:
            ngram_list = [tuple(s[i:i+n]) for s in sents for i in range(len(s)-n+1)]       
        ngram_set = set(ngram_list)
        ngram_dict = Counter(ngram_list) 
        return ngram_set, ngram_dict
    
    def ngram_prob(self, ngram, num_words, unigram_dic, bigram_dic, trigram_dic):
        prob = None
        if len(ngram) == 1:
            prob = unigram_dic[ngram[0]]/num_words
        elif len(ngram) == 2:
            prob = bigram_dic[(ngram[0],ngram[1])]/unigram_dic[ngram[0]]
        elif len(ngram) == 3:
            prob =  trigram_dic[(ngram[0], ngram[1], ngram[2])]/bigram_dic[(ngram[0], ngram[1])]
        else:
            raise ValueError
        return prob
    
    def add_k_smoothing_ngram(self, ngram, k, num_words, unigram_dic, bigram_dic, trigram_dic):
        s_prob = None
        V = len(unigram_dic)
        num_sentences = self.unigram_dict[self.STOP]
        if len(ngram) == 1:
            s_prob = (unigram_dic[ngram[0]] + k)/(num_words + k*V)
        elif len(ngram) == 2:
            if ngram[0] == self.START:
                s_prob = (bigram_dic[(ngram[0], ngram[1])] + k)/(num_sentences + k*V)
                #s_prob = (bigram_dic[(ngram[0], ngram[1])] + k)/(0 + k*V)
            else:
                s_prob = (bigram_dic[(ngram[0], ngram[1])] + k)/(unigram_dic[ngram[0]] + k*V)
        elif len(ngram) == 3:
            if ngram[0:2] == [self.START,self.START]:
                s_prob = (trigram_dic[(ngram[0], ngram[1], ngram[2])] + k)/(num_sentences + k*V)
                #s_prob = (trigram_dic[(ngram[0], ngram[1], ngram[2])] + k)/(0 + k*V)
            else:
                s_prob = (trigram_dic[(ngram[0], ngram[1], ngram[2])] + k)/(bigram_dic[(ngram[0], ngram[1])] + k*V)
        else:
            raise ValueError
        return s_prob
    
    def interpolation_ngram(self, ngram, lam, num_words, unigram_dic, bigram_dic, trigram_dic):
        s_prob = None
        num_sentences = self.unigram_dict[self.STOP]
        if len(ngram) == 1:
            s_prob = lam[2]*unigram_dic[ngram[0]]/num_words
        elif len(ngram) == 2:
            if ngram[0] == self.START:
                s_prob = lam[1]*(bigram_dic[(ngram[0], ngram[1])])/num_sentences + lam[2]*unigram_dic[ngram[1]]/num_words
            else:
                s_prob = lam[1]*(bigram_dic[(ngram[0], ngram[1])])/(unigram_dic[ngram[0]]) + lam[2]*unigram_dic[ngram[1]]/num_words
        elif len(ngram) == 3:
            if ngram[0:2] == [self.START,self.START]:
                tri_pro = (trigram_dic[(ngram[0], ngram[1], ngram[2])])/num_sentences
                #tri_pro = 0
                bi_pro = (bigram_dic[(ngram[1], ngram[2])])/num_sentences
                #bi_pro = 0
            else:
                if bigram_dic[(ngram[0], ngram[1])] == 0:
                    tri_pro = 0
                    bi_pro = (bigram_dic[(ngram[1], ngram[2])])/(unigram_dic[ngram[1]])
                else:
                    tri_pro = (trigram_dic[(ngram[0], ngram[1], ngram[2])])/(bigram_dic[(ngram[0], ngram[1])])
                    bi_pro = (bigram_dic[(ngram[1], ngram[2])])/(unigram_dic[ngram[1]])
                    
            s_prob = lam[0]*tri_pro + lam[1]*bi_pro + lam[2]*unigram_dic[ngram[2]]/num_words
        else:
            raise ValueError
        return s_prob
    
    def rank_docs(self, query, contexts):
        query_parsed = self.query_ngrams(query, 2)
        prob_dict = {}
        for id, context in enumerate(contexts):
            unigram_set, unigram_dict = self.compute_ngram([context], 1)
            bigram_set, bigram_dict = self.compute_ngram([context], 2)
            trigram_set, trigram_dict = self.compute_ngram([context], 3)
            num_words = sum([v for _,v in unigram_dict.items()])
            prob = 1
            for ngram in query_parsed:
                # print(ngram, bigram_dict)
                prob *= self.ngram_prob(ngram, num_words,unigram_dict, bigram_dict, trigram_dict)
            prob_dict[id] = prob
        prob_dict_sorted = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        return prob_dict_sorted

