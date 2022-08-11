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
    # valid = dataset['validation']
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
    # dataset = load_dataset("squad_v2")
    # valid = dataset['validation']
    # method = 'doc2vec'
    # vsm_index = VSMIndex(method, valid)
    # vsm = VSM(vsm_index)
    # allvsm, tagged_sorted_dict = vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=3)
    # vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=3)
    # pprint(tagged_sorted_dict)
    
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
    valid = dataset['validation']
    bm25_index = BM25Index(valid)
    ngram = Ngram(bm25_index)
    
    unigram_list = []
    bigram_list= []
    trigram_list = []
    
    for id, context in enumerate(bm25_index.contexts):
        unigram_set, unigram_dict = ngram.compute_ngram([context], 1)
        
        bigram_set, bigram_dict = ngram.compute_ngram([context], 2)
        
        trigram_set, trigram_dict = ngram.compute_ngram([context], 3)
        unigram_list.append(unigram_dict)
        trigram_list.append(trigram_dict)
        bigram_list.append(bigram_dict)
    # ngram_scores, ngram_contexts = ngram.score_docs("duck you when", 3, alpha=None, top_k=10, expand_query=True, verbose=False)
    # print(unigram_list[0], f" \n\n\n {unigram_list[1]}")
    eval = Eval(valid, bm25_index.contexts, language_model=ngram)

    print(eval.average_rank(top_k=len(bm25_index.contexts), expand_query=False, verbose=False, k_arg=1, alpha=2, uni_dict_list=unigram_list, bi_dict_list=bigram_list, tri_dict_list=trigram_list))
    # main idea: Get the top K docs, fetch the docs's query tagging
    # if tagging exist for the doc then yes
    # eval = Eval()
    # eval.precision_at_top_k()
