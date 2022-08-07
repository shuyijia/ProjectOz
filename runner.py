from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex

if __name__ == "__main__":
    
    # BM25
    # dataset = load_dataset("squad_v2")
    # valid = dataset['train']
    # bm25_index = BM25Index(valid)
    # bm25 = BM25(bm25_index)
    # query = "In what country is Normandy located"
    # bm25_scores = bm25.score_docs(query, print_top_k=10, expand_query=False)

    # VSM
    dataset = load_dataset("squad_v2")
    valid = dataset['train']
    method = 'tfidf'
    vsm_index = VSMIndex(method, valid)

    # convert a query to vectorized form
    query = 'what is phonology'
    vectorized_query = vsm_index.infer(query)

    vsm = VSM(vsm_index)
    vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=10)
    # vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=10)
