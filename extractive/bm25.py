from datasets import load_dataset
from tqdm import tqdm
import math
import numpy as np

from helpers.utils import preprocess
from query_expansion.expand_query import get_expanded_query


class BM25:
    def __init__(self, bm25_index):
        self.index = bm25_index.index
        self.docs = bm25_index.contexts
        self.average_doc_len = np.mean(np.array([len(doc) for doc in self.docs]))
        
    def score_doc(self, query, doc, k1, b):
        score = 0.0
        
        for term in query:
            if doc not in self.index.get(term, {}):
                continue
            tf_term = self.index[term][doc]
            df_term = len(self.index[term])

            log_term = math.log(len(self.docs)/df_term)
            numerator = (k1+1) * tf_term
            denominator = k1 * ((1-b) +b * (len(self.docs[doc])/self.average_doc_len)) + tf_term
            score += log_term * (numerator/denominator)
        return score
    
    def score_docs(self, query, k1=1.5, b=0.75, print_top_k=0, expand_query=True):
        if expand_query:
            query = get_expanded_query(query)
        print("THE QUERY IS", query)
        query = preprocess([query])[0]
        bm25_scores = {}
        for doc_id in tqdm(range(len(self.docs))):
            bm25_scores[doc_id] = self.score_doc(query,doc_id, k1, b)
        bm25_scores_sorted =  dict(sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True))

        print_doc_count = 0
        for doc_id in bm25_scores_sorted:
            if print_doc_count == print_top_k:
                break
            print(f"Doc Rank: {print_doc_count +1}\nDoc score for Doc {doc_id}: {bm25_scores_sorted[doc_id]} \n\nWords in Doc {doc_id}: {' '.join(self.docs[doc_id])}", end=f"\n\n{'*'*175}\n\n")
            print_doc_count += 1
        return bm25_scores_sorted
            
    