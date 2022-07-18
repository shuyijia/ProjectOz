from datasets import load_dataset
from extractive.bm25 import BM25
from index.bm25_index import BM25Index

if __name__ == "__main__":
    
    dataset = load_dataset("squad_v2")
    valid = dataset['validation']
    bm25_index = BM25Index(valid)
    bm25 = BM25(bm25_index)
    query = "In what country is Normandy located"
    bm25_scores = bm25.score_docs(query, print_top_k=10)
