from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex
from helpers.utils import preprocess, punctuation_removal
from extractive.ngram import Ngram
from eval.eval import Eval

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
    vectorized_query = vsm_index.infer(query)

    vsm = VSM(vsm_index)
    # vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=3)
    vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=3)

    method = 'tfidf'
    vsm_index = VSMIndex(method, valid)

    # convert a query to vectorized form
    query = 'what is phonology'
    vectorized_query = vsm_index.infer(query)

    vsm = VSM(vsm_index)
    # vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=10)
    # vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=10)
    dataset = load_dataset("squad_v2")
    valid = dataset['train']
    data_preprocessed = preprocess(valid['context'])
    print(len(data_preprocessed))
    ngram = Ngram()
    # print(ngram.rank_docs("hi how are you", data_preprocessed))
    print(valid)

    # main idea: Get the top K docs, fetch the docs's query tagging
    # if tagging exist for the doc then yes
    # eval = Eval()
    # eval.precision_at_top_k()
