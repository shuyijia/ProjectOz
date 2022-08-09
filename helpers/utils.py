import re

# preprocess methods
def punctuation_removal(s):
  return re.sub(r'[^\w\s]', '', s)

def case_folding(s, case='lower'):
  return s.lower() if case == 'lower' else s.upper()

def split_and_remove_spaces(s):
  return [w for w in s.split(' ') if w != '']

def preprocess(l, remove_punctuation=True, fold_cases=True):
  if remove_punctuation:
    l = map(punctuation_removal, l)
  if fold_cases:
    l = map(case_folding, l)

  return list(map(split_and_remove_spaces, l))

# def preprocess(l, remove_punctuation=True, fold_cases=True):
#   if remove_punctuation:
#     l = map(punctuation_removal, l)
#   if fold_cases:
#     l = map(case_folding, l)

#   # out = []
#   # for i in l:
#   #   out.append(split_and_remove_spaces(i))

#   # print("PREPROICESDSEDEDED")
#   # print(out[:3])
#   # print("asdasdfsdf")
#   #print(list(map(split_and_remove_spaces, l))[:7])

#   return list(map(split_and_remove_spaces, l))