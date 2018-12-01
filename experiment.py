#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import embed
sns.set()

from nltk.corpus import wordnet as wn
from sklearn.manifold import TSNE
from gensim.models.poincare import PoincareModel, ReconstructionEvaluation


# ==================================================
# Calculate Poincare and Euclidean Distance Ratio
# ==================================================

def poincare_dist(u, v, eps=1e-5):
    boundary = 1 - eps
    u_norm_square = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
    v_norm_square = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
    square_dist = torch.sum(torch.pow(u - v, 2), dim=-1)
    x = square_dist / ((1 - u_norm_square) * (1 - v_norm_square)) * 2 + 1
    # arcosh
    z = torch.sqrt(torch.pow(x, 2) - 1)
    return torch.log(x + z)


def euclidean_dist(u, v):
    return torch.sqrt(torch.sum(torch.pow(u - v, 2), dim=-1))


# plot distance ratio
def plot_distance_ratio():
    origin = torch.Tensor([0, 0])
    euclidean_ratios = []
    poincare_ratios = []
    x_norm = []
    for i in np.linspace(1e-3, 0.7, 5000):
        x = torch.Tensor([i, i])
        y = torch.Tensor([i, -i])
        x_norm.append(x.norm())
        euclidean_ratio = euclidean_dist(x, y) / (euclidean_dist(origin, x) + euclidean_dist(origin, y))
        poincare_ratio = poincare_dist(x, y) / (poincare_dist(origin, x) + poincare_dist(origin, y))
        euclidean_ratios.append(np.round(euclidean_ratio.item(), 4))
        poincare_ratios.append(np.round(poincare_ratio.item(), 4))

    ax = sns.lineplot(x_norm, euclidean_ratios, label='Euclidean Distance Ratio')
    ax = sns.lineplot(x_norm, poincare_ratios, label='Poincare Distance Ratio')
    ax.set_xlabel('X Norm')
    ax.set_ylabel('Distance Ratio')

    x = [i for i in np.linspace(1e-3, 0.7, 8)]
    y = [-i for i in np.linspace(1e-3, 0.7, 8)]

    sns.scatterplot(x, x, label='Point A')
    sns.scatterplot(x, y, label='Point B')
    sns.scatterplot([0], [0], color='black', label='Origin')
    ax.save_figure('distance_ratio.png')


# ========================================
# Utility functions to get WordNet data
# ========================================

def get_relations_from_file(file):
    """Get pairs of hypernonym-hyponynm relations data from input file."""
    relations = []
    with open(file) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            hypo, hyper = line.split()
            relations.append([hypo, hyper])

    return relations


def get_all_hyponyms(synset):
    hypo = lambda s: s.hyponyms()

    return list(synset.clousure(hypo))


def get_synset_hypos_subtree_names(word):
    hypo = lambda s: s.hyponyms()

    synset = wn.synsets(word)[0]
    synset_hypos_subtree = list(synset.closure(hypo))  # all descendents
    synset_hypos_subtree_names = list(map(lambda x: x.name(), synset_hypos_subtree))

    return synset_hypos_subtree_names


# ===================================
# Poincare and Euclidean Embedding
# ===================================

class Args:

    def __init__(self, dim, dset, fout, distfn, lr, epochs, batchsize, negs, nproc, ndproc, eval_each, burnin, debug):
        self.dim = dim
        self.dset = dset
        self.fout = fout
        self.distfn = distfn
        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize
        self.negs = negs
        self.nproc = nproc
        self.ndproc = ndproc
        self.eval_each = eval_each
        self.burnin = burnin
        self.debug = debug
        self.retraction = None
        self.rgrad = None


def train_embedding(dim, dset, fout, distfn, lr=0.3, epochs=200, batchsize=50, negs=20,
                       nproc=5, ndproc=2, eval_each=10, burnin=20, debug=False):
    opt = Args(dim, dset, fout, distfn, lr, epochs, batchsize, negs, nproc, ndproc, eval_each, burnin, debug)
    embed.main(opt)


def gensim_poincare_embedding(relations_data, embedding_size, negative_samples=20, epochs=50,
                              print_every=5, output=None):
    model = PoincareModel(relations_data, size=embedding_size, negative=negative_samples)
    model.train(epochs=epochs, print_every=print_every)

    # save model
    if output:
        model.save(output)

    return model


class EvaluateEmbedding(ReconstructionEvaluation):
    """Evaluate reconstruction for trained embedding (model not trained with gensim implementation)

    Subclass of ReconstructionEvaluation from gensim implementation.
    """

    def __init__(self, data, embeddingFile):
        """Initialize evaluation instance with tsv file containing relation pairs and embedding to be evaluated."""
        items = set()
        embedding = torch.load(embeddingFile)
        embedding_vocab = embedding['objects']
        relations = defaultdict(set)
        for pair in data:
            assert len(pair) == 2, 'Hypernym pair has more than two items'
            item_1_index = embedding_vocab.index(pair[0])
            item_2_index = embedding_vocab.index(pair[1])
            relations[item_1_index].add(item_2_index)
            items.update([item_1_index, item_2_index])

        self.items = items
        self.relations = relations
        self.embedding = embedding

    def evaluate_mean_rank_and_map(self, max_n=None):
        """Evaluate mean rank and MAP for reconstruction.
        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.
        Returns
        -------
        (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).
        """
        ranks = []
        avg_precision_scores = []
        for i, item in enumerate(self.items, start=1):
            if item not in self.relations:
                continue
            item_relations = list(self.relations[item])
            other_idx = np.where(np.arange(1181) != item)
            item_vector = self.embedding['model']['lt.weight'][[item]]
            other_vectors = self.embedding['model']['lt.weight'][other_idx]
            item_distances = self.vector_distance_batch(item_vector,
                                                        other_vectors)  # compute distances between node and all nodes
            positive_relation_ranks, avg_precision = \
                self.get_positive_relation_ranks_and_avg_prec(item_distances, item_relations)
            ranks += positive_relation_ranks
            avg_precision_scores.append(avg_precision)
            if max_n is not None and i > max_n:
                break
        return np.mean(ranks), np.mean(avg_precision_scores)

    @staticmethod
    def vector_distance_batch(vector_1, vectors_all):
        euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
        return euclidean_dists


# ==========================================
# Plot embedding in 2 dimension using TSNE
# ==========================================

def tsne_plot(gensim_model, synset_hypos_subtree=None, random_label=False, outFile=None):
    n_vocab = len(gensim_model.kv.vocab)
    labels = []
    vectors = []
    colors = [0] * n_vocab

    for idx, word in enumerate(gensim_model.kv.vocab):
        vectors.append(gensim_model.kv[word])
        labels.append(word)
        # for coloring one particular hyponyms subtree
        if synset_hypos_subtree:
            if word in synset_hypos_subtree:
                colors[idx] = 1

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    fig = plt.figure(figsize=(16, 16))
    plt.scatter(x, y, c=colors, cmap='rainbow')
    if random_label:
        for i in np.random.choice(len(x), 15, replace=False):
            plt.scatter(x[i], y[i], c='green', cmap='rainbow')
            plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom')

    # Save figure
    if outFile:
        fig.savefig(outFile)


# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', help='Embedding dimension', type=int)
    parser.add_argument('--poincare', help='Train poincare embedding', action='store_true')
    parser.add_argument('--euclidean', help='Train euclidean embedding', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate embedding', action='store_true')
    parser.add_argument('--plot', help='Plot poincare embedding', action='store_true')
    args = parser.parse_args()

    embeddingDim = args.dim
    inputFile = 'wordnet/mammal_closure.tsv'
    outFilePoincare = 'mammal-poincare-' + str(embeddingDim) + '.pth'
    outFileEuclidean = 'mammal-euclidean-' + str(embeddingDim) + '.pth'

    # train poincare embedding
    if args.poincare:
        train_embedding(embeddingDim, inputFile, outFilePoincare, 'poincare')
    # train euclidean embedding
    if args.euclidean:
        train_embedding(embeddingDim, inputFile, outFileEuclidean, 'euclidean')

    # Evaluate embedding using gensim implementation
    mammal_relations = get_relations_from_file(inputFile)
    print('Evaluating poincare embedding')
    poincare_results = EvaluateEmbedding(mammal_relations, outFilePoincare).evaluate()
    print(poincare_results)

    print('Evaluating euclidean embedding')
    euclidean_results = EvaluateEmbedding(mammal_relations, outFileEuclidean).evaluate()
    print(euclidean_results)

    # Plot poincare embedding
    if args.plot:
        print('getting relations data')
        mammal_relations = get_relations_from_file(inputFile)
        print('training model with gensim')
        gensim_model = gensim_poincare_embedding(mammal_relations, embeddingDim)
        print('plot embedding')
        dog_subtree = get_synset_hypos_subtree_names('dog')
        tsne_plot(gensim_model, dog_subtree, random_label=True, outFile='tsne-dog.png')