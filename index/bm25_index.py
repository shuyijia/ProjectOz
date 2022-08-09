import numpy as np
from helpers.utils import preprocess, punctuation_removal
from datasets import load_dataset
from ordered_set import OrderedSet

class BM25Index:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.contexts = preprocess(list(OrderedSet(self.dataset['context'])))

        self.build()

    def build(self):
        '''
        build an index for term frequency and document frequency
        word -> {doc1 : 10, doc2 : 3, ..., doc_n : 23}
        '''
        index = {}

        for i, doc in enumerate(self.contexts):
            for term in doc:
                if term not in index:
                    index[term] = {}
                if i not in index[term]:
                    index[term][i] = 0
                index[term][i] += 1
        
        self.index = index

if __name__ == "__main__":
    datasets = load_dataset("squad_v2")
    valid = datasets['validation']
    idx = BM25Index(valid)

    print("number of unique docs: ", len(idx.contexts))

    word = 'hydrogen'
    print('tf in all docs containing {}: {}'.format(word, idx.index[word]))
    print('df of {}: {}'.format(word, len(idx.index[word])))