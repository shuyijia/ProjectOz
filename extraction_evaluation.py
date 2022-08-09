from datasets import load_dataset
from extractive.bm25 import BM25
from extractive.vsm import VSM
from index.bm25_index import BM25Index
from index.vsm_index import VSMIndex
from helpers.utils import preprocess, punctuation_removal
from extractive.ngram import Ngram
from eval.eval import Eval
from ordered_set import OrderedSet
from prettyprinter import pprint
from tqdm import tqdm

dataset = load_dataset("squad_v2")
valid = dataset['validation']
method = 'doc2vec'
vsm_index = VSMIndex(method, valid)

vsm = VSM(vsm_index)
eval = Eval(valid, vsm_index.contexts, vsm)