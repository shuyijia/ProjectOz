import nltk
from query_expansion.part_of_speech_tag import *
from query_expansion.preprocessing import *

def download_nltk_packages():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    
download_nltk_packages()

def generate_tokens(sentence):
    tokens = tokenizer(sentence)
    tokens = pos_tagger(tokens)
    tokens = stopword_treatment(tokens)
    synsets = get_synsets(tokens)
    synonyms = get_tokens_from_synsets(synsets)
    synonyms = underscore_replacer(synonyms)
    hypernyms = get_hypernyms(synsets)
    hypernyms = get_tokens_from_hypernyms(hypernyms)
    hypernyms = underscore_replacer(hypernyms)
    tokens = {**synonyms, **hypernyms}
    return tokens

def get_expanded_query(query, expand_terms):
    query_tokens=generate_tokens(" ".join(expand_terms))
    return f"{' '.join(query)} {' '.join(list(query_tokens.keys()))}"
