from collections import Counter
import itertools
import numpy as np
import math

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

 
class Ngram:
    
    def calculate_interpolated_sentence_probability(self, sentence, doc, collection, alpha=0.75, normalize_probability=True, smoothing=True):
        '''
        calculate interpolated sentence/query probability using both sentence and collection unigram models.
        sentence: input sentence/query
        doc: unigram language model a doc. HINT: this can be an instance of the UnigramLanguageModel class
        collection: unigram language model a collection. HINT: this can be an instance of the UnigramLanguageModel class
        alpha: the hyperparameter to combine the two probability scores coming from the document and collection language models.
        normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
        '''
        
        doc_unigram_model = UnigramLanguageModel(doc, smoothing=smoothing)
        collection_unigram_model = UnigramLanguageModel(collection, smoothing=smoothing)

        interpolated_sentence_prob = 1
        for term in sentence:
            if term == SENTENCE_START or term == SENTENCE_END:
                continue
            interpolated_sentence_prob*= ((alpha*doc_unigram_model.calculate_unigram_probability(term)) + (1-alpha)*collection_unigram_model.calculate_unigram_probability(term))
        return interpolated_sentence_prob if normalize_probability else math.log(interpolated_sentence_prob,2)

    def compute_ngram(self, sents, n):

        ngram_set = None
        ngram_dict = None

        if n == 1:
            ngram_list = [s[i:i+n] for s in sents for i in range(len(s)-n+1)]
            ngram_list = list(itertools.chain(*ngram_list))
        elif n >= 2:
            ngram_list = [tuple(s[i:i+n]) for s in sents for i in range(len(s)-n+1)]       
        ngram_set = set(ngram_list)
        ngram_dict = Counter(ngram_list)

        return ngram_set, ngram_dict


ngram = Ngram()
print(ngram.compute_ngram([["hi", "hello","how", "are", "you", "today"]], 2))

