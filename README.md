## Poincare Embedding
Investigate the paper: [Poincare Embedding for Learning Hierarchical Representation] (https://arxiv.org/pdf/1705.08039.pdf)

### Packages
- numpy
- matplotlib
- seaborn
- pytorch
- sklearn
- nltk
- gensim

### Usage
#### Data: `wordnet/mammal_closure.tsv`
#### Train embedding
```
# use poincare distance
python experiment.py --dim [embedding dimension] --poincare

# use euclidean distance
python experiment.py --dim [embedding dimension] --euclidean
```

#### Evaluate embedding
```
python experiment.py --dim [embedding dimension] --evaluate
```

#### Plot poincare embedding
```
python experiment.py --dim [embedding dimension] --plot
```
