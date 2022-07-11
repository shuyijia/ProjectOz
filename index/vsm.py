from datasets import load_dataset
import numpy as np
from utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


class VSMIndex:
    def __init__(self, method, dataset, vector_size=40):
        self.method = method
        self.dataset = dataset
        self.vector_size = vector_size
        self.contexts = preprocess(list(set(self.dataset['context'])))

        if self.method == 'doc2vec':
            self.gen_doc2vec(vector_size=vector_size)
        elif self.method == 'tfidf':
            self.gen_tfidf()
        else:
            raise ValueError('Unsppported method type.')

    def gen_doc2vec(self, vector_size=40, min_count=2, epochs=30):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.contexts)]
        self.model = Doc2Vec(documents, vector_size=vector_size, min_count=min_count, epochs=epochs)

        # use trained doc2vec to represent docs
        self.doc_vecs = np.zeros((len(self.contexts), vector_size))
        for i, doc in enumerate(self.contexts):
            self.doc_vecs[i, :] = self.model.infer_vector(doc)

    def gen_tfidf(self, max_features=500):
        documents = [' '.join(doc) for doc in self.contexts]
        self.model = TfidfVectorizer(max_features=max_features)
        self.X = self.model.fit_transform(documents)

        self.doc_vecs = np.zeros((len(self.contexts), max_features))
        for i, doc in enumerate(documents):
            self.doc_vecs[i, :] = self.model.transform([doc]).todense()

    def infer(self, doc):
        '''
        doc: list of words
        '''
        if self.method == 'doc2vec':
            return self.model.infer_vector(doc)
        elif self.method == 'tfidf':
            return self.model.transform(' '.join(doc))


if __name__ == '__main__':
    datasets = load_dataset("squad_v2")
    valid = datasets['validation']

    method = 'doc2vec'

    idx = VSMIndex(method, valid)

    # get all vectorzied documents
    all_docs = idx.doc_vecs

    # convert a query to vectorized form
    query = 'what is the name of the last French king'.split(' ')
    vectorized_query = idx.infer(query)
    print(vectorized_query.shape)