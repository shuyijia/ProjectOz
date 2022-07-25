from datasets import load_dataset
from index.vsm_index import VSMIndex
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import jaccard_score 

class VSM:
    def __init__(self, vsm_index):
        self.vsm_index = vsm_index
        #self.index = vsm_index.index
        self.docs = vsm_index.contexts
        self.all_docs_vec = vsm_index.doc_vecs
        self.average_doc_len = np.mean(np.array([len(doc) for doc in self.docs]))
        self.vsm_method = "NIL"

    def vsm(self, query, vsm_method = "cosine_similarity", print_top_k=0):
        self.vectorized_query = self.vsm_index.infer(query)
        factor = 5
        self.vsm_method = vsm_method
        doc_index = 0

        if vsm_method == "cosine_similarity":
            all_vsm = {}
            for doc in self.all_docs_vec:
                a = self.vectorized_query
                b = doc.reshape(1, doc.shape[0])
                all_vsm[doc_index] = cosine_similarity(a, b)[0][0]
                doc_index += 1
            vsm_scores_sorted = self.score_docs(all_vsm, print_top_k)
            return all_vsm
        elif vsm_method == "jaccard_similarity":
            all_vsm = {}
            for doc in self.all_docs_vec:
                a = np.rint(np.multiply(self.vectorized_query.reshape(self.vectorized_query.shape[1]), factor))
                b = np.rint(np.multiply(doc, factor))
                j_score = jaccard_score(a, b, average="micro")
                all_vsm[doc_index] = j_score
                doc_index += 1
            vsm_scores_sorted = self.score_docs(all_vsm, print_top_k)
            return all_vsm
        else:
            print("Method doesn't exist!")
            return None
        
    def score_docs(self, all_vsm, print_top_k=0):
        vsm_scores_sorted =  dict(sorted(all_vsm.items(), key=lambda item: item[1], reverse=True))
        print_doc_count = 0
        print("VSM Method: ", self.vsm_method)
        for doc_id in vsm_scores_sorted:
            if print_doc_count == print_top_k:
                break
            print(f"Doc Rank: {print_doc_count +1}\nDoc score for Doc {doc_id}: {vsm_scores_sorted[doc_id]} \nWords in Doc {doc_id}: {' '.join(self.docs[doc_id])}", end=f"\n\n{'*'*175}\n")
            print_doc_count += 1
        print("\n"*5)
        return vsm_scores_sorted

if __name__ == "__main__":
    dataset = load_dataset("squad_v2")
    valid = dataset['validation']
    method = 'tfidf'
    vsm_index = VSMIndex(method, valid)

    # convert a query to vectorized form
    query = 'what is the name of the last French king'
    vectorized_query = vsm_index.infer(query)

    vsm = VSM(vsm_index)
    vsm.vsm(query, vsm_method="cosine_similarity", print_top_k=3)
    vsm.vsm(query, vsm_method="jaccard_similarity", print_top_k=3)


    

    