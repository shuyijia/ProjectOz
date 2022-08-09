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
    
    # # BM25
    # dataset = load_dataset("squad_v2")
    # valid = dataset['train']
    # bm25_index = BM25Index(valid)
    # bm25 = BM25(bm25_index)
    # query = "In what country is Normandy located"
    # bm25_scores, doc_contexts = bm25.score_docs(query, top_k=3, expand_query=False)
    # print(bm25_scores)
    # VSM
    # dataset = load_dataset("squad_v2")
    # valid = dataset['train']
    # bm25_index = BM25Index(valid)
    # bm25 = BM25(bm25_index)
    # query = "In what country is Normandy located"
    # bm25_scores, doc_contexts = bm25.score_docs(query, top_k=3, expand_query=False)

    # print(bm25_scores)

    # VSM
    dataset = load_dataset("squad_v2")
    valid = dataset['validation']
    method = 'tfidf'
    vsm_index = VSMIndex(method, valid)

    # convert a query to vectorized form
    query = 'Who did the Irish culture have a profound effect on?'
    print("Query: ", query, '\n')
    vectorized_query = vsm_index.infer(query)

    vsm = VSM(vsm_index)
    # allvsm, tagged_sorted_dict = vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=3)
    # vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=3)
    # pprint(tagged_sorted_dict)

    eval = Eval(valid, vsm_index.contexts, vsm)

    
    # method = 'tfidf'
    # vsm_index = VSMIndex(method, valid)

    # # convert a query to vectorized form
    # query = 'what is phonology'
    # vectorized_query = vsm_index.infer(query)

    # vsm = VSM(vsm_index)
    # vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=10)
    # vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=10)
    
    # NGRAM
    # dataset = load_dataset("squad_v2")
    # valid = dataset['train']
    # data_preprocessed = preprocess(list(OrderedSet(valid['context'])))
    # print(len(data_preprocessed))
    # ngram = Ngram()
    # print(ngram.rank_docs("hi how are you", data_preprocessed))
    
    
    # main idea: Get the top K docs, fetch the docs's query tagging
    # if tagging exist for the doc then yes
 
    dataset = load_dataset("squad_v2")
    valid = dataset['train']
    bm25_index = BM25Index(valid)
    ngram = Ngram(bm25_index)
    ngram_scores, ngram_contexts = ngram.score_docs("duck you when", 3, alpha=None, top_k=10, expand_query=True, verbose=False)

    # main idea: Get the top K docs, fetch the docs's query tagging
    # if tagging exist for the doc then yes
    # eval = Eval()
    # eval.precision_at_top_k()
