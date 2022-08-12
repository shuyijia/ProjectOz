import random

from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from extractive.ngram import Ngram
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex
from generative.squad_qna_pred import *
from generative.squad_qna_topk import custom_predict

def extractive(query, method='bm25', k=10):
    dataset = load_dataset("squad_v2")
    print("Method: {}".format(method))

    if method == 'bm25':
        bm25_index = BM25Index(dataset['train'])
        bm25 = BM25(bm25_index)

        scores, contexts = bm25.score_docs(query, top_k=k, expand_query=False)
    
    elif method == 'ngram':
        bm25_index = BM25Index(dataset['train'])
        ngram = Ngram(bm25_index)

        scores, contexts = ngram.score_docs(query, 2, alpha=[0.45, 0.45, 0.1], top_k=k, expand_query=False)

    elif method == 'vsm':
        vsmindex = VSMIndex(method='tfidf', dataset=dataset['train'])
        vsm = VSM(vsm_index=vsmindex)
        scores, contexts = vsm.vsm(query, vsm_method='cosine_similarity', print_top_k=k)
        scores = dict(list(scores.items())[: k])
        contexts = contexts[:k]
    
    return list(scores.keys()), contexts


if __name__ == "__main__":

    dataset = load_dataset("squad_v2")
    idx = random.randint(0, len(dataset['train'])-1)
    query = dataset['train'][idx]['question']
    ans = dataset['train'][idx]['answers']['text']

    id_list, context_list = extractive(query, method='bm25', k=10)
    question_list = [query] * len(id_list)

    pred_head = Squad_QnA_Prediction()

    print("=======")
    out = custom_predict(id_list, context_list, question_list, pred_head)
    for k, v in out.items():
        print(k, v)
    print("=======")
    print("Query idx: ", idx)
    print(query, ans)