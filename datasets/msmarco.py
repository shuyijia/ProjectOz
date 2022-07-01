import pandas as pd
import re, time
from collections import Counter

class Dictionary:
    def __init__(self) -> None:
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class DataBank:
    def __init__(self, case_folding=True, remove_punct=True, dataset=None):
        self.case_folding = case_folding
        self.remove_punct = remove_punct
        self.dataset = dataset
        if not self.dataset:
            return Exception("dataset is None")
        self.dictionary = Dictionary()

        self.id_list = [x['query_id'] for x in self.dataset]
        self.content_list =[''.join(x['passages']['passage_text']) for x in self.dataset]

        # # preprocess
        self.preprocess()
        self.id2contents = dict(zip(self.id_list, self.content_list))

        # # build bank
        self.build_bank()
    
    def preprocess(self):
        # remove punctuations
        if self.remove_punct:
            self.content_list = map(self.punctuation_removal, self.content_list)
        # case folding
        if self.case_folding:
            self.content_list = map(self.fold_cases, self.content_list)
        
        # split
        self.content_list = list(map(self.split_and_remove_spaces, self.content_list))

    def build_bank(self):
        int_texts = {}
        w_doc_freq = {}

        for q_id, str_tokens in self.id2contents.items():
            int_tokens = []
            for tk in str_tokens:
                self.dictionary.add_word(tk)
                int_tk = self.dictionary.word2idx[tk]
                int_tokens.append(int_tk)

            int_texts[q_id] = int_tokens

            counter = Counter(int_tokens)
            for tk, cnt in counter.items():
                if tk not in w_doc_freq:
                    w_doc_freq[tk] = {}
                w_doc_freq[tk][q_id] = cnt

        self.int_texts = int_texts
        self.w_doc_freq = w_doc_freq

    def punctuation_removal(self, s):
        return re.sub(r'[^\w\s]', '', s)
    
    def fold_cases(self, s, case='lower'):
        return s.lower() if case == 'lower' else s.upper()

    def split_and_remove_spaces(self, s):
        return [w for w in s.split(' ') if w != '']
    
    def get_str_vec(self, int_vec):
        return list(map(lambda x: self.dictionary.idx2word[x], int_vec))

    def q2vec(self, q):
        new_q = q
        if self.remove_punct:
            new_q = self.punctuation_removal(new_q)
        if self.case_folding:
            new_q = self.fold_cases(new_q)
        
        new_q = self.split_and_remove_spaces(new_q)
        return list(map(lambda x: self.dictionary.word2idx[x], new_q))