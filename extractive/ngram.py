from collections import Counter
import itertools
import numpy as np
import math

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

 
class Ngram:
    def __init__(self, training_sents):
        self.START = '<START>'
        self.STOP = '<STOP>'
        self.training_sents = training_sents
        self.unigram_dict = self.compute_ngram(self.training_sents)
        
    def compute_ngram(self, sents, n):

        ngram_set = None
        ngram_dict = None

        if n == 1:
            ngram_list = [s[i:i+n] for s in sents for i in range(len(s)-n+1)]
            ngram_list = list(itertools.chain(*ngram_list))
        elif n >= 2:
            ngram_list = [tuple(s[i:i+n]) for s in sents for i in range(len(s)-n+1)]       
        ngram_set = set(ngram_list)
        ngram_dict = Counter(ngram_list)

        return ngram_set, ngram_dict

    ###################################
    def pad_sents(self,sents, n):
        '''
        Pad the sents according to n.
        params:
            sents: list[list[str]] --- list of sentences.
            n: int --- specify the padding type, 1-gram, 2-gram, or 3-gram.
        return:
            padded_sents: list[list[str]] --- list of padded sentences.
        '''
        #padded_sents = None
        padded_sents = []
        ### YOUR CODE HERE
        if n == 1:
            padded_sents = [s+[self.STOP] for s in sents]
        elif n == 2:
            padded_sents = [[self.START]+s+[self.STOP] for s in sents]
        elif n == 3:
            padded_sents = [[self.START]*2 + s +  [self.STOP] for s in sents]
        else:
            raise ValueError
        ### END OF YOUR CODE
        return padded_sents
    
    def ngram_prob(self, ngram, num_words, unigram_dic, bigram_dic, trigram_dic):
        '''
        params:
            ngram: list[str] --- a list that represents n-gram
            num_words: int --- total number of words
            unigram_dic: dict{ngram: counts} --- a dictionary that maps each 1-gram to its number of occurences in "sents";
            bigram_dic: dict{ngram: counts} --- a dictionary that maps each 2-gram to its number of occurence in "sents";
            trigram_dic: dict{ngram: counts} --- a dictionary that maps each 3-gram to its number occurence in "sents";
        return:
            prob: float --- probability of the "ngram"
        '''
        prob = None
        ### YOUR CODE HERE
        if len(ngram) == 1:
            prob = unigram_dic[ngram[0]]/num_words
        elif len(ngram) == 2:
            prob = bigram_dic[(ngram[0],ngram[1])]/unigram_dic[ngram[0]]
        elif len(ngram) == 3:
            prob =  trigram_dic[(ngram[0], ngram[1], ngram[2])]/bigram_dic[(ngram[0], ngram[1])]
        else:
            raise ValueError
        ### END OF YOUR CODE
        return prob
    
    def add_k_smoothing_ngram(self, ngram, k, num_words, unigram_dic, bigram_dic, trigram_dic):
        '''
        params:
            ngram: list[str] --- a list that represents n-gram
            k: float 
            num_words: int --- total number of words
            unigram_dic: dict{ngram: counts} --- a dictionary that maps each 1-gram to its number of occurences in "sents";
            bigram_dic: dict{ngram: counts} --- a dictionary that maps each 2-gram to its number of occurence in "sents";
            trigram_dic: dict{ngram: counts} --- a dictionary that maps each 3-gram to its number occurence in "sents";
        return:
            s_prob: float --- probability of the "ngram"
        '''
        s_prob = None
        V = len(unigram_dic)
        ### YOUR CODE HERE
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
        ### END OF YOUR CODE
        return s_prob
    
    def interpolation_ngram(self, ngram, lam, num_words, unigram_dic, bigram_dic, trigram_dic):
        '''
        params:
            ngram: list[str] --- a list that represents n-gram
            lam: list[float] --- a list of length 3.lam[0], lam[1] and lam[2] are correspondence to trigram, bigram and unigram,repectively.
                                If len(ngram) == 1, lam[0]=lam[1]=0, lam[2]=1. If len(ngram) == 2, lam[0]=0. lam[0]+lam[1]+lam[2] = 1.
            num_words: int --- total number of words
            unigram_dic: dict{ngram: counts} --- a dictionary that maps each 1-gram to its number of occurences in "sents";
            bigram_dic: dict{ngram: counts} --- a dictionary that maps each 2-gram to its number of occurence in "sents";
            trigram_dic: dict{ngram: counts} --- a dictionary that maps each 3-gram to its number occurence in "sents";
        return:
            s_prob: float --- probability of the "ngram"
        '''
        s_prob = None
        ### YOUR CODE HERE
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
        ### END OF YOUR CODE
        return s_prob


ngram = Ngram()
print(ngram.compute_ngram([["hi", "hello","how", "are", "you", "today"]], 2))

