# Fangorn
SUTD 50.045 Information Retrieval Project 

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