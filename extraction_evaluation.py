from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex
from helpers.utils import preprocess, punctuation_removal
from extractive.ngram import Ngram
from eval.eval import Eval
from ordered_set import OrderedSet
from prettyprinter import pprint
from tqdm import tqdm


dataset = load_dataset("squad_v2")
valid = dataset['validation']

#############################################################################################
# # BM25 Evaluation
# bm25_index = BM25Index(valid)
# bm25 = BM25(bm25_index)
# eval = Eval(valid, bm25_index.contexts, bm25=bm25)
# eval.average_rank(top_k=len(bm25_index.contexts), expand_query=True, verbose=False)
#############################################################################################
# # VSM Evaluation
# '''
# 2 Vectorization Method Options:
# 1. 'doc2vec'
# 2. 'tfidf'
# '''
# method = 'doc2vec'
# vsm_index = VSMIndex(method, valid)
# vsm = VSM(vsm_index)

# '''
# 2 Similarity Functions Options:
# 1. "cosine_similarity"
# 2. "jaccard_similarity"
# '''
# eval = Eval(valid, vsm_index.contexts, vsm=vsm)
# eval.average_rank(vsm_method="cosine_similarity", top_k=10)
# #eval.average_rank(vsm_method="jaccard_similarity", top_k=10)
#############################################################################################
# # NGRAM
# '''
# main idea: Get the top K docs, fetch the docs's query tagging
# if tagging exist for the doc then yes
# '''
# bm25_index = BM25Index(valid)
# ngram = Ngram(bm25_index)

# unigram_list = []
# bigram_list= []
# trigram_list = []

# for id, context in enumerate(bm25_index.contexts):
#     unigram_set, unigram_dict = ngram.compute_ngram([context], 1)
#     bigram_set, bigram_dict = ngram.compute_ngram([context], 2)
#     trigram_set, trigram_dict = ngram.compute_ngram([context], 3)
#     unigram_list.append(unigram_dict)
#     trigram_list.append(trigram_dict)
#     bigram_list.append(bigram_dict)

# eval = Eval(valid, bm25_index.contexts, language_model=ngram)
# print(eval.average_rank(top_k=len(bm25_index.contexts), expand_query=False, verbose=False, k_arg=1, alpha=2, uni_dict_list=unigram_list, bi_dict_list=bigram_list, tri_dict_list=trigram_list))
#############################################################################################