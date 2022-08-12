# Project Oz
A Question Answering Information Retrieval System
SUTD's 50.045 Information Retrieval Module Project

Project Oz aims to implement a span extraction information retrieval system for question
answering using the Stanford Question Answering Dataset (SQuAD). Our implementation of
Okapi-BM25 outperformed other extractive models like vector space model and language model.
Project Oz’s end to end span extraction pipeline clearly demonstrated ability to extract exact
answers for a given query.

For more informtation about the 50.045 Information Retrieval Course, visit
https://istd.sutd.edu.sg/undergraduate/courses/50045-information-retrieval

Done by:
- [Anirudh Shrinivason](https://github.com/Anirudh181001)
- [Jia Shuyi](https://github.com/shuyijia)
- [Pheh Jing Jie](https://github.com/jjbecomespheh)

# Overview framework
![overall_framework](https://user-images.githubusercontent.com/50895766/184366226-b675c8dd-743a-487b-ab34-60cd34769bcc.png)

# Getting Started
```
pip install -r requirements.txt
```

Load SQuAD dataset:

```python
from datasets import load_dataset
datasets = load_dataset("squad_v2")
```
# What have been implemented
- Preprocessing
  - Punctuation removal
  - Case folding
  - Remove white spaces
  - Tokenization
- Vector Space Model (VSM)
  - Vectorization Techniques:
    - TF-IDF
    - Doc2Vec
  - Similarity Functions:
    - Cosine Similarity
    - Jaccard Similarity
- Okapi-BM25
- Language Model
  - n-gram Language Model
  - Laplace Smoothing
  - Interpolated n-gram Model
- Span Extraction using Transformer

# How to run the models
To run extractive models, uncomment desired models and run 
```
runner.py
```

To evaluate extractive model, uncomment desired models and run
```
extraction_evaluation.py
```

To use end to end span extraction model, run
```
end2end.py
```


