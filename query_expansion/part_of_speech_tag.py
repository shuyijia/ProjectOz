import nltk
from nltk.corpus import wordnet as wn
from query_expansion.constants import POS_TAG_MAP
import re

def pos_tag_converter(pos_tag_map, nltk_pos_tag):
    root_tag = nltk_pos_tag[0:2]
    try:
        pos_tag_map[root_tag]
        return pos_tag_map[root_tag]
    except KeyError:
        return ''
    
def get_synsets(tokens):
    synsets = []
    for token in tokens:
        wn_pos_tag = pos_tag_converter(POS_TAG_MAP,token[1])
        if wn_pos_tag == '':
            continue
        else:
            synsets.append(wn.synsets(token[0], wn_pos_tag))
    return synsets

def get_tokens_from_synsets(synsets):
    tokens = {}
    for synset in synsets:
        for s in synset:
            if s.name() in tokens:
                tokens[s.name().split('.')[0]] += 1
            else:
                tokens[s.name().split('.')[0]] = 1
    return tokens


def get_hypernyms(synsets):
    hypernyms = []
    for synset in synsets:
        for s in synset:
            hypernyms.append(s.hypernyms())
            
    return hypernyms

def get_tokens_from_hypernyms(synsets):
    tokens = {}
    for _ in synsets:
        for s in synsets:
            for ss in s:
                if ss.name().split('.')[0] in tokens:
                    tokens[(ss.name().split('.')[0])] += 1
                else:
                    tokens[(ss.name().split('.')[0])] = 1
    return tokens

def underscore_replacer(tokens):
    new_tokens = {}
    for key in tokens.keys():
        mod_key = re.sub(r'_', ' ', key)
        new_tokens[mod_key] = tokens[key]
    return new_tokens