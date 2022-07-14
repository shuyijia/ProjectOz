from datasets import load_dataset
import numpy as np
from index.utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class VSMIndex:
    def __init__(self, method, dataset, vector_size=40):
        self.method = method
        self.dataset = dataset
        self.vector_size = vector_size
        self.contexts = preprocess(list(set(self.dataset['context'])))

        if self.method == 'doc2vec':
            self.gen_doc2vec(vector_size=vector_size)
        elif self.method == 'tfidf':
            self.gen_tfidf(vector_size=vector_size)
        else:
            raise ValueError('Unsppported method type.')

    def gen_doc2vec(self, vector_size=40, min_count=2, epochs=30):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.contexts)]
        self.model = Doc2Vec(documents, vector_size=vector_size, min_count=min_count, epochs=epochs)

        # use trained doc2vec to represent docs
        self.doc_vecs = np.zeros((len(self.contexts), vector_size))
        for i, doc in enumerate(self.contexts):
            self.doc_vecs[i, :] = self.model.infer_vector(doc)

    def gen_tfidf(self, vector_size=40, max_features=500):
        documents = [' '.join(doc) for doc in self.contexts]
        self.model = TfidfVectorizer(max_features=max_features)
        self.X = self.model.fit_transform(documents)

        self.doc_vecs = np.zeros((len(self.contexts), max_features))
        for i, doc in enumerate(documents):
            self.doc_vecs[i, :] = self.model.transform([doc]).todense()

        self.svd = TruncatedSVD(n_components=vector_size, n_iter=7, random_state=42)
        self.doc_vecs = self.svd.fit_transform(self.doc_vecs)

    def infer(self, doc):
        '''
        doc: str
        '''
        if self.method == 'doc2vec':
            doc = doc.split(' ')
            return self.model.infer_vector(doc, steps=20, alpha=0.025)
        elif self.method == 'tfidf':
            sparse = self.model.transform([doc])
            return self.svd.transform(sparse)

if __name__ == '__main__':
    datasets = load_dataset("squad_v2")
    valid = datasets['validation']

    method = 'tfidf'
    idx = VSMIndex(method, valid)

    # get all vectorzied documents
    all_docs = idx.doc_vecs
    print(type(all_docs))
    print(all_docs.shape)
    print(all_docs)

    # convert a query to vectorized form
    query = 'what is the name of the last French king'
    vectorized_query = idx.infer(query)
    print(vectorized_query.shape)