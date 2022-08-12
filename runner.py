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

if __name__ == "__main__":
    
    query = "What is the name of Harry Potter's owl?"
    k = 3
    dataset = load_dataset("squad_v2")
    valid = dataset['validation']
    verbose = True # True to print all the retrieved documents

    if verbose:
        print("Query: ", query, '\n')
    #############################################################################################
    # # BM25
    # bm25_index = BM25Index(valid)
    # bm25 = BM25(bm25_index)
    # bm25_scores, doc_contexts = bm25.score_docs(query, top_k=k, expand_query=True, verbose=verbose)
    #############################################################################################
    # # VSM
    # '''
    # 2 Vectorization Method Options:
    # 1. 'doc2vec'
    # 2. 'tfidf'
    # '''
    # method = 'doc2vec'
    # vsm_index = VSMIndex(method, valid)
    # vsm = VSM(vsm_index)
    # vectorized_query = vsm_index.infer(query) # convert a query to vectorized form
    # '''
    # 2 Similarity Functions Options:
    # 1. "cosine_similarity"
    # 2. "jaccard_similarity"
    # '''
    # allvsm, tagged_sorted_dict = vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=k, verbose=verbose)
    #############################################################################################
    # NGRAM
    # To use NGAM, need to also use BM25 Index
    bm25_index = BM25Index(valid)

    data_preprocessed = preprocess(list(OrderedSet(valid['context'])))
    print(len(data_preprocessed))
    ngram = Ngram(bm25_index)
    print(ngram.score_docs("hi how are you", data_preprocessed, verbose=verbose, expand_query=False))
    #############################################################################################
    


