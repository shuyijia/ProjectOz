# Project Oz
A Question Answering Information Retrieval System
SUTD's 50.045 Information Retrieval Module Project

For more informtation about the 50.045 Information Retrieval Course, visit
https://istd.sutd.edu.sg/undergraduate/courses/50045-information-retrieval

Done by:
- [Anirudh Shrinivason](https://github.com/Anirudh181001)
- [Jia Shuyi](https://github.com/shuyijia)
- [Pheh Jing Jie](https://github.com/jjbecomespheh)

# Getting Started
```
pip install -r requirements.txt
```

Load SQuAD dataset:

```python
from datasets import load_dataset
datasets = load_dataset("squad_v2")
```
# How to run the models
To run extractive models, use 
```
runner.py
```

To evaluate extractive model, use 
```
extraction_evaluation.py
```

To use end to end span extraction model, use
```
end2end.py
```