# Fangorn
SUTD 50.045 Information Retrieval Project

- [Anirudh Shrinivason](https://github.com/Anirudh181001)
- [Jia Shuyi](https://github.com/shuyijia)
- [Pheh Jing Jie](https://github.com/jjbecomespheh)

# Getting Started
Install HuggingFace Datasets Hub:

```
pip install datasets
```

Load MS Marco V1.1 dataset:

```python
from datasets import load_dataset
dataset = load_dataset("ms_marco", "v1.1")
```

Load SQuAD dataset:

```python
from datasets import load_dataset
datasets = load_dataset("squad_v2")
```