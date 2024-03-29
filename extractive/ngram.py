from collections import Counter
import itertools
import numpy as np
import math
from datasets import load_dataset
from ordered_set import OrderedSet
from helpers.utils import preprocess
from query_expansion.expand_query import get_expanded_query
# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

 
class Ngram:
    def __init__(self, bm25_index):
        self.START = '<START>'
        self.STOP = '<STOP>'
        self.unigram_dict = None
        self.index = bm25_index.index
        self.docs = bm25_index.contexts
        self.average_doc_len = np.mean(np.array([len(doc) for doc in self.docs]))
        
    def expand_query_idf(self, query, top_k=2):
        idf_query_terms = {}
        for term in query:
            if term not in self.index:
                 continue 
            idf_query_terms[term] = (self.get_idf(self.get_df(term)))
        idf_query_terms = dict(sorted(idf_query_terms.items(), key=lambda item: item[1], reverse=True))
        return get_expanded_query(query, list(idf_query_terms.keys())[:top_k])
        
    def get_idf(self, df_term):
        return math.log(len(self.docs)/df_term, 10)
    
    def get_df(self, term):
        return len(self.index[term]) 
    
    def query_ngrams(self, query, n, expand_query=False):
        query = preprocess([query])[0]
        if expand_query:
            query = self.expand_query_idf(query)
            query = preprocess([query])[0]
        if n ==1:
            return [(q, ) for q in query]
        elif n == 2:
            return [query[i:i+2] for i in range(len(query)-1)]
        elif n == 3:
            return [query[i:i+3] for i in range(len(query)-2)]
        
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
            try:
                prob = unigram_dic[ngram[0]]/num_words
            except ZeroDivisionError:
                prob = 1/len(unigram_dic)
        elif len(ngram) == 2:
            try:
                prob = bigram_dic[(ngram[0],ngram[1])]/unigram_dic[ngram[0]]
            except ZeroDivisionError:
                prob = 1/len(unigram_dic)
        elif len(ngram) == 3:
            try:
                prob =  trigram_dic[(ngram[0], ngram[1], ngram[2])]/bigram_dic[(ngram[0], ngram[1])]
            except ZeroDivisionError:
                prob = 1/len(unigram_dic)
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
        if len(ngram) == 1:
            try:
                s_prob = lam[2]*unigram_dic[ngram[0]]/num_words
            except ZeroDivisionError:
                s_prob = 1/len(unigram_dic)
        elif len(ngram) == 2:
            try:
                s_prob = lam[1]*(bigram_dic[(ngram[0], ngram[1])])/(unigram_dic[ngram[0]]) + lam[2]*unigram_dic[ngram[1]]/num_words
            except ZeroDivisionError:
                s_prob = 1/len(unigram_dic)
        elif len(ngram) == 3:
            if bigram_dic[(ngram[0], ngram[1])] == 0:
                try:
                    tri_pro = 0
                    bi_pro = (bigram_dic[(ngram[1], ngram[2])])/(unigram_dic[ngram[1]])
                except ZeroDivisionError:
                    bi_pro = 1/len(unigram_dic)
            else:
                try:
                    tri_pro = (trigram_dic[(ngram[0], ngram[1], ngram[2])])/(bigram_dic[(ngram[0], ngram[1])])
                    bi_pro = (bigram_dic[(ngram[1], ngram[2])])/(unigram_dic[ngram[1]])
                except ZeroDivisionError:
                    tri_pro = bi_pro = 1/len(unigram_dic)
                    
            s_prob = lam[0]*tri_pro + lam[1]*bi_pro + lam[2]*unigram_dic[ngram[2]]/num_words
        else:
            raise ValueError
        return s_prob
    
    def score_docs(self, query, k, alpha=None, top_k = 0, expand_query=False, verbose=False, unigram_dict=None, bigram_dict = None, trigram_dict=None):
        query_parsed = self.query_ngrams(query, k, expand_query=expand_query)
        prob_dict = {}
        u_passed = b_passed = t_passed = False
        if unigram_dict: u_passed=True
        if trigram_dict: t_passed=True
        if bigram_dict: b_passed=True
        
        if u_passed and b_passed and t_passed:
            for id, context in enumerate(self.docs):
                num_words = sum([v for _,v in unigram_dict[id].items()])
                prob = 1
                for ngram in query_parsed:
                    if alpha is None:
                        prob *= self.ngram_prob(ngram, num_words,unigram_dict[id], bigram_dict[id], trigram_dict[id])
                    elif type(alpha) == int:
                        prob *= self.add_k_smoothing_ngram(ngram, alpha, num_words,unigram_dict[id], bigram_dict[id], trigram_dict[id])
                    elif type(alpha) == list:
                        prob *= self.interpolation_ngram(ngram, alpha, num_words,unigram_dict[id], bigram_dict[id], trigram_dict[id])
                prob_dict[id] = prob
        else:
            for id, context in enumerate(self.docs):
                if not u_passed:
                    unigram_set, unigram_dict = self.compute_ngram([context], 1)
                if not b_passed:
                    bigram_set, bigram_dict = self.compute_ngram([context], 2)
                if not t_passed:
                    trigram_set, trigram_dict = self.compute_ngram([context], 3)
                num_words = sum([v for _,v in unigram_dict.items()])
                prob = 1
                for ngram in query_parsed:
                    if alpha is None:
                        prob *= self.ngram_prob(ngram, num_words,unigram_dict, bigram_dict, trigram_dict)
                    elif type(alpha) == int:
                        prob *= self.add_k_smoothing_ngram(ngram, alpha, num_words,unigram_dict, bigram_dict, trigram_dict)
                    elif type(alpha) == list:
                        prob *= self.interpolation_ngram(ngram, alpha, num_words,unigram_dict, bigram_dict, trigram_dict)
                prob_dict[id] = prob
        prob_dict_sorted = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        prob_dict_sorted_top_k = dict(list(prob_dict_sorted.items())[: top_k])
        print_doc_count = 0
        doc_contexts = []
        for doc_id in prob_dict_sorted_top_k:
            if verbose:
                print(f"Doc Rank: {print_doc_count +1}\nDoc score for Doc {doc_id}: {prob_dict_sorted[doc_id]} \n\nWords in Doc {doc_id}: {' '.join(self.docs[doc_id])}", end=f"\n\n{'*'*175}\n\n")
            doc_contexts.append(' '.join(self.docs[doc_id]))
            print_doc_count += 1
        return prob_dict_sorted_top_k, doc_contexts

