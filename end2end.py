from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex
from generative.squad_qna_pred import *
from generative.squad_qna_topk import custom_predict

def extractive(query, method='bm25'):
    dataset = load_dataset("squad_v2")

    if method == 'bm25':
        print("Method: {}".format(method))
        bm25_index = BM25Index(dataset['train'])
        bm25 = BM25(bm25_index)

        bm25_scores, doc_contexts = bm25.score_docs(query, top_k=10, expand_query=False)
        return list(bm25_scores.keys()), doc_contexts


if __name__ == "__main__":
    query = 'In what country is Normandy located'
    id_list, context_list = extractive(query)
    question_list = [query] * len(id_list)

    pred_head = Squad_QnA_Prediction()

    out = custom_predict(id_list, context_list, question_list, pred_head)
    print(out)