from datasets import load_dataset
import numpy as np
from helpers.utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from prettyprinter import pprint
from extractive.vsm import VSM
from helpers.utils import *
import time

class Eval():
    def __init__(self, dataset, processed_context, vsm):
        self.dataset = dataset
        self.processed_context = processed_context
        # self.index_alignment()
        self.vsm = vsm
        self.average_rank()

    def average_rank(self):
        '''
        out:    key => doc_id
                val => query
        '''
        ranks = []
        for i, each in enumerate(self.dataset):
            q = each['question']
            doc = each['context']
            doc_id = self.processed_context.index(preprocess([doc])[0])
            tic = time.time()
            allvsm, tagged_sorted_dict = self.vsm.vsm(q, vsm_method = "cosine_similarity", print_top_k=10)
            toc = time.time()
            print((toc-tic) /1000)
            ranks.append(list(tagged_sorted_dict.keys()).index(doc_id))
            if i % 100 == 0:
                print(i)
        print("Done", np.mean(ranks))
        return np.mean(ranks)


    def index_alignment(self):
        length_of_dataset = len(self.dataset['question'])
        index = 0
        self.questiions_id_dict = {}
        self.questiions_id_dict[0] = [self.dataset['question'][0]]

        for i in range(1, length_of_dataset): #300): #
            if self.dataset['context'][i] == self.dataset['context'][i-1]:
                if index not in self.questiions_id_dict:
                    self.questiions_id_dict[index] = [self.dataset['question'][i]]
                else:
                    self.questiions_id_dict[index].append(self.dataset['question'][i])
            else:
                index += 1
                if index not in self.questiions_id_dict:
                    self.questiions_id_dict[index] = [self.dataset['question'][i]]
                else:
                    self.questiions_id_dict[index].append(self.dataset['question'][i])
            
            if i % 1000 == 0:
                print("Eval: ", i)

        # pprint(self.questiions_id_dict)
        # self.check_questions_doc_tagging(337)
        

    def check_questions_doc_tagging(self, x):
        print("Questions-Doc Index: ", x)
        for i in range(len(self.questiions_id_dict[x])):
            print("Question ", i,": ", self.questiions_id_dict[x][i])
        print("\nProcessed Context: ", " ".join(self.processed_context[x]))


    def better_than_average_rank(self, x):
        pass






    def precision_at_k(self, r, k):
        """Score is precision @ k (This we solve for you!)

        Relevance is binary (nonzero is relevant).

        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)

        Returns:
            Precision @ k

        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def precision_at_top_k(self, query, docs, k):
        """Score is precision @ top k"""
       
        # Compute precision top k
        # precision_top_k = len(list_detected_fraudulent_transactions) / top_k
        
        # return list_detected_fraudulent_transactions, precision_top_k
        pass

    def average_precision(self, r):
        """Score is average precision (area under PR curve)

        Relevance is binary (nonzero is relevant).

        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)

        Returns:
            Average precision
        """
        #write your code here
        try:
            delta_r = 1. / sum(r)
        except ZeroDivisionError:
            #print("You can't divide by zero!")
            return 0
        avg_p = sum([self.precision_at_k(r, x + 1) * delta_r for x, y in enumerate(r) if y])
        return avg_p

    def mean_average_precision(self, rs):
        """Score is mean average precision

        Relevance is binary (nonzero is relevant).

        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)

        Returns:
            Mean average precision
        """
        #write your code here
        m_avg_p = np.mean([self.average_precision(r) for r in rs])
        return m_avg_p

    def dcg_at_k(self, r, k, method=0):
        """Score is discounted cumulative gain (dcg)

        Relevance is positive real values.  Can use binary
        as the previous methods.

        Example from
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
      
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
            k: Number of results to consider
            method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                    If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

        Returns:
            Discounted cumulative gain
        """

        #write your code here (return appropriate value)
        r = np.asfarray(r)[:k]
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def ndcg_at_k(self, r, k, method=0):
        """Score is normalized discounted cumulative gain (ndcg)

        Relevance is positive real values.  Can use binary
        as the previous methods.

        Example from
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
            k: Number of results to consider
            method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                    If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

        Returns:
            Normalized discounted cumulative gain
        """
        #write your code here (return appropriate value)
        dcg = self.dcg_at_k(r, k, method=method)
        idcg = self.dcg_at_k(sorted(r, reverse=True), k, method=method)
        if idcg:
            return dcg/idcg
        return 0.

if __name__ == "__main__":

    # print(average_precision([0,1,0,0,1,0,1,0,0,0]))
    # print(average_precision([1,0,1,0,0,1,0,0,1,1]))
    # print(mean_average_precision([[1,0,1,0,0,1,0,0,1,1],[0,1,0,0,1,0,1,0,0,0]]))
    # print(dcg_at_k([1.0,0.6,0.0,0.8,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.2,0.0],3))
    # print(ndcg_at_k([1.0,0.6,0.0,0.8,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.2,0.0],3))
    pass
