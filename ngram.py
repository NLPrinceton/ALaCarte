# evaluates n-gram embeddings on classification tasks
# to run use command:
#   python ALaCarte/ngram.py $TASKS $VECTORFILE $DIMENSION $WORDCOOC $NGRAMCOOC
# where:
#   TASKS is a list of comma-delimited tasks to be evaluated (e.g. sst,sst_fine,imdb)
#   VECTORFILE is the Amazon embeddings (can get vectors from http://nlp.cs.princeton.edu/ALaCarte/vectors)
#   DIMENSION is the word embedding dimension
#   WORDCOOC is the output of running ALaCarte/cooc.py for words on Amazon corpus (can get corpora from http://nlp.cs.princeton.edu/ALaCarte/corpora)
#   NGRAMCOOC is a space-delimited list of outputs of running ALaCarte/cooc.py for n-grams on Amazon corpus (e.g. /n/fs/nlpdatasets/ALaCache/Amazon_2Gram /n/fs/nlpdatasets/ALaCache/Amazon_3Gram)

import sys
import nltk
import numpy as np
from sklearn.preprocessing import normalize
from ALaCarte.compute import *
from ALaCarte.cooc import *
from text_embedding.documents import *
from text_embedding.features import *
from text_embedding.vectors import *


# NOTE: can be memory-intensive on larger tasks (e.g. IMDB); decrease MAXSLICE if necessary
MAXSLICE = 1000000


def alabong(A, word_embeddings, lists, coocs, counts):
  n = len(lists)
  def represent(documents):
    output = []
    docs = tokenize(doc.lower() for doc in documents)
    for k, kgramlist, kgramcooc, kgramcount in zip(range(1, n+1), lists, coocs, counts):
      kgrams = [list(nltk.ngrams(doc, k)) for doc in docs]
      vocab = {kgram for doc in kgrams for kgram in doc}
      where = np.array([i for i, kgram in enumerate(kgramlist) if kgram in vocab and kgramcount[i]])
      bong = docs2bofs(kgrams, vocabulary=kgramlist, format='csc')
      output.append(np.zeros((len(documents), word_embeddings.shape[1]), dtype=FLOAT))
      for offset in range(0, where.shape[0], MAXSLICE):
        indices = where[offset:offset+MAXSLICE]
        if k > 1:
          vecs = normalize(A.predict(kgramcooc[indices].dot(word_embeddings)/kgramcount[indices,None])) / k
        else:
          vecs = normalize(word_embeddings[indices])
        output[-1] += bong[:,indices].dot(vecs)
    return np.hstack(output)
  return represent, None, True


if __name__ == '__main__':

  tasks = sys.argv[1].split(',')
  vectorfile = sys.argv[2]
  dimension = int(sys.argv[3])
  write('\rLoading Word, Word Cooccurrences')
  word_cooc, wordlist, word_counts = alacache(sys.argv[4])
  write('\rLoading Word Embeddings to Matrix')
  word_embeddings = vocab2mat(wordlist, vectorfile=vectorfile, dimension=dimension, unit=False)
  write('\rComputing Linear Context Transform')
  A = linear_transform(word_cooc, word_embeddings, word_counts, weights=word_counts>999, fit_intercept=False, n_jobs=-1)
  n = 1

  coocs = [None]
  lists = [[(word,) for word in wordlist]]
  counts = [word_counts]
  for root in sys.argv[5:]:
    write('\rLoading '+root+20*' ')
    cooc, _, _, ngramlist, ngram_counts = alacache(root)
    coocs.append(cooc)
    lists.append(ngramlist)
    counts.append(ngram_counts)
    n += 1
  
  write('\rComputing N-Gram Feature Embeddings'+10*' '+'\n')
  represent, prepare, invariant = alabong(A, word_embeddings, lists, coocs, counts)
  for task in sys.argv[1].split(','):
    evaluate(task, represent, prepare=prepare, invariant=invariant, verbose=True, intercept=task in TASKMAP['pairwise task'], n_jobs=1)
