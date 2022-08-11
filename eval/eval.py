from datasets import load_dataset
import numpy as np
from helpers.utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from prettyprinter import pprint
from extractive.vsm import VSM
from helpers.utils import *
import time
from tqdm import tqdm

class Eval():
    def __init__(self, dataset, processed_context, vsm=None, bm25=None, language_model=None):
        self.dataset = dataset
        self.processed_context = processed_context
        # self.index_alignment()
        self.vsm = vsm
        self.bm25 = bm25
        self.language_model = language_model
        # self.average_rank()

    def average_rank(self, vsm_method=None, top_k=None, expand_query=None, verbose=None, k_arg=None, alpha=None, uni_dict_list=None, bi_dict_list=None, tri_dict_list=None):
        '''
        out:    key => doc_id
                val => query
        '''
        ranks = []
        # print(type(k_arg))
        for i, each in tqdm(enumerate(self.dataset)):
            q = each['question']
            doc = each['context']
            doc_id = self.processed_context.index(preprocess([doc])[0])

            if self.vsm:
                _, tagged_sorted_dict = self.vsm.vsm(q, vsm_method = vsm_method, print_top_k=top_k)
            elif self.bm25:
                tagged_sorted_dict, _ = self.bm25.score_docs(q, top_k=top_k, expand_query=expand_query, verbose=verbose)
            elif self.language_model:
                tagged_sorted_dict, _ = self.language_model.score_docs(q, k=k_arg, alpha=alpha, top_k = top_k, expand_query=expand_query, verbose=verbose, unigram_dict=uni_dict_list[doc_id], bigram_dict=bi_dict_list[doc_id], trigram_dict=tri_dict_list[doc_id])

            ranks.append(list(tagged_sorted_dict.keys()).index(doc_id))
            
            if i%1000 == 0:
                print(f"i: {i} Mean so far:{np.mean(ranks)}")
        print("Done", np.mean(ranks))
        return np.mean(ranks)
