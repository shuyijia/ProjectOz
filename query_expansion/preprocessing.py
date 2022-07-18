from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

def tokenizer(sentence):
    return word_tokenize(sentence)

def pos_tagger(tokens):
    return nltk.pos_tag(tokens)

def stopword_treatment(tokens):
    stopword = stopwords.words('english')
    result = []
    for token in tokens:
        if token[0].lower() not in stopword:
            result.append(tuple([token[0].lower(), token[1]]))
    return result
