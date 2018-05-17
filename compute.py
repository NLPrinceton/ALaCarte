import sys
from collections import Counter
from collections import defaultdict
from itertools import chain
import nltk
import numpy as np
from scipy import sparse as sp
from sklearn.linear_model import LinearRegression as LR


FLOAT = np.float32
INT = np.int32


def ngram_context(strdoc, intdoc, vocabulary, n=1, wndo2=5, unkgram=None):
  '''sliding window around n-grams in a document
  Args:
    strdoc: list of tokens (as strings)
    intdoc: list of indices (as ints); len(intdoc) == len(strdoc)
    vocabulary: n-gram vocabulary (set of n-grams or dict with n-grams as keys)
    n: n in n-gram
    wndo2: half the window size
    unkgram: map n-grams not in vocabulary to this n-gram; if None does not yield such n-grams
  Returns:
    (n-gram, int generator) generator over (n-gram, context window pairs)
  '''

  wndo2pn = wndo2+n
  unk = not unkgram is None
  for i, ngram in enumerate(nltk.ngrams(strdoc, n)):
    if ngram in vocabulary:
      yield ngram, chain(intdoc[max(i-wndo2, 0):i], intdoc[i+n:i+wndo2pn])
    elif unk:
      yield unkgram, chain(intdoc[max(i-wndo2, 0):i], intdoc[i+n:i+wndo2pn])


def counts2mat(featcoocs, featlist, shape, dtype):
  '''computes matrix from feature-word cooccurrence counts
  Args:
    featcoocs: dict mapping features to Counters
    featlist: list of features
    shape: matrix shape
    dtype: dtype of matrix
  Returns:
    sparse matrix in CSR format
  '''

  rows, cols, values = zip(*((i, j, count) for i, feat in enumerate(featlist) for j, count in featcoocs[feat].items()))
  return sp.coo_matrix((values, (rows, cols)), shape=shape, dtype=dtype).tocsr()


def cooc_matrix(corpus, featlist, wordlist, doc2wnd=ngram_context, unk=None, overlap=False, avg=False, wei=False, interval=1000000, verbose=False, comm=None, **kwargs):
  '''constructs feature, word cooccurrence matrix
  Args:
    corpus: iterable of lists of strings
    featlist: list of hashable features
    wordlist: list of strings
    doc2wnd: takes list of tokens, list of indices, and set of features and returns a (feature, index iterable) generator
    unk: map words not in wordlist to this token (must be in wordlist); if None excludes OOV words
    overlap: if True subtracts feature count from cooccurrence of feature with any word it contains; features must be iterable
    avg: uses average over window size rather than cooccurrence counts
    wei: weight co-occurring words by distance from window
    interval: number of documents between conversion to sparse matrix
    verbose: write context matrix construction progress
    comm: MPI Communicator; outputs are None for non-root processes
    kwargs: passed to doc2wnd
  Returns:
    cooccurrence matrix in CSR format, vector of feature counts, vector of word counts
  '''

  assert not (overlap and (avg or wei)), "correcting for overlap not compatible with averaging or weighting"

  featset = set(featlist)
  featcounts = Counter()
  F = len(featlist)
  unki = -1 if unk is None else wordlist.index(unk)
  word2index = {word: i for i, word in enumerate(wordlist)}
  wordcounts = Counter()
  V = len(wordlist)

  rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
  write = lambda msg: sys.stdout.write(msg) and sys.stdout.flush()
  dtype = FLOAT if (avg or wei) else INT
  if not rank:
    matrix = sp.csr_matrix((F, V), dtype=dtype)
  featcoocs = defaultdict(lambda: Counter())

  for i, doc in enumerate(corpus):
    if i%size == rank:
      indices = [word2index.get(word, unki) for word in doc]
      wordcounts.update(indices)
      if avg:
        for feat, window in doc2wnd(doc, indices, featset, **kwargs):
          window = list(window)
          if window:
            increment = 1.0/len(window)
            cooccounts = featcoocs[feat]
            for index in window:
              cooccounts[index] += increment
          featcounts[feat] += 1
      elif wei:
        for feat, window in doc2wnd(doc, indices, featset, **kwargs):
          window = list(window)
          if window:
            length = len(window)
            half = int(length/2)
            recip = 1.0/length
            cooccounts = featcoocs[feat]
            for j, index in enumerate(window[:half]):
              cooccounts[index] += recip/(half-j)
            for j, index in enumerate(window[half:]):
              cooccounts[index] += recip/(j+1)
          featcounts[feat] += 1
      else:
        for feat, window in doc2wnd(doc, indices, featset, **kwargs):
          featcoocs[feat].update(window)
          featcounts[feat] += 1
    if not (i+1)%interval:
      if rank:
        comm.send(counts2mat(featcoocs, featlist, (F, V), dtype), dest=0)
      else:
        matrix += sum((comm.recv(source=j) for j in range(1, size)), counts2mat(featcoocs, featlist, (F, V), dtype))
        if verbose:
          write('\rProcessed '+str(i+1)+' Documents; Sparsity: '+str(matrix.nnz)+'/'+str(F*V)+'; Coverage: '+str((matrix.sum(1)>0).sum())+'/'+str(F))
      featcoocs = defaultdict(lambda: Counter())

  if size > 1:
    featcounts = comm.reduce(featcounts, root=0)
    wordcounts = comm.reduce(wordcounts, root=0)
  if rank:
    comm.send(counts2mat(featcoocs, featlist, (F, V), dtype), dest=0)
    return 3*[None]
  matrix += sum((comm.recv(source=j) for j in range(1, size)), counts2mat(featcoocs, featlist, (F, V), dtype))

  if overlap:
    for feat, coocs in featcoocs.items():
      count = featcounts[feat]
      for word in feat:
        index = word2index.get(word)
        if not index is None:
          coocs[index] -= count
  if verbose:
    write('\rProcessed '+str(i+1)+' Documents; Sparsity: '+str(matrix.nnz)+'/'+str(F*V)+'; Coverage: '+str((matrix.sum(1)>0).sum())+'/'+str(F)+'\n')
  return matrix, np.array([featcounts[feat] for feat in featlist], dtype=INT), np.array([wordcounts[word2index[word]] for word in wordlist], dtype=INT)


def symmetric_cooc_matrix(corpus, wordlist, unk=None, **kwargs):
  '''constructs symmetric word, word cooccurrence matrix
  Args:
    corpus: iterable of lists of strings
    wordlist: list of strings
    unk: map words not in wordlist to this token (must be in wordlist); if None excludes OOV words
    kwargs: passed to cooc_matrix
  Returns:
    cooccurrence matrix in CSR format, vector of word counts
  '''

  unkgram = None if unk is None else (unk,)
  return cooc_matrix(corpus, [(word,) for word in wordlist], wordlist, unk=unk, n=1, unkgram=unkgram, **kwargs)[:2]


def linear_transform(cooc_matrix, word_embeddings, word_counts, Regression=LR, weights=None, **kwargs):
  '''learns linear transform from context vectors to original embeddings
  Args:
    cooc_matrix: cooccurrence matrix of size (V, V)
    word_embeddings: embedding matrix of size (V, d)
    word_counts: word count vector of length V
    Regression: regression class (from sklearn.linear_model)
    weights: sample weight vector of length V; ignored if None
    kwargs: passed to Regression
  Returns:
    fitted Regression object
  '''

  select = word_counts > 0
  if not weights is None:
    select *= weights > 0
    weights = weights[select]

  return Regression(**kwargs).fit(cooc_matrix[select].dot(word_embeddings) / word_counts[select,None], word_embeddings[select], weights)
