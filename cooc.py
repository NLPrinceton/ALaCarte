# computes (feature, word) cooccurrence matrices given text corpora
# to run use command:
#   mpirun -n 8 python ALaCarte/cooc.py $FEATURE $CORPUSFILE $VOCABFILE $OUTPUTROOT $N
# where:
#   FEATURE is one of ngram, synset, word
#   CORPUSFILE is the corpus textfile (skipped if $FEATURE==synset)
#   VOCABFILE is the word vocabulary
#   OUTPUTROOT is the output name (script will make files $OUTPUTROOT.npz and $OUTPUTROOT.pkl)
#   N is a positive integer (skipped if not $FEATURE==ngram)
# if $FEATURE==ngram: computes cooccurrences for all n-grams in sst,sst_fine,imdb,mr,cr,subj,mpqa,trec,mrpc,sick tasks
# if $FEATURE==synset: computes cooccurrences for all synsets in SemCor

import pickle
import sys
from itertools import chain
import nltk
import numpy as np
from nltk.corpus import semcor
from scipy import sparse as sp
from ALaCarte.compute import *
from text_embedding.documents import *
from text_embedding.features import *


UNK = '<unk>'


def ngram_vocab(n):
  ngrams = lambda docs: {ngram for doc in tokenize(doc.lower() for doc in docs) for ngram in nltk.ngrams(doc, n)}
  return sorted(set.union(*(ngrams(sst_fine(partition)[0]) for partition in ['train', 'test'])))
  vocabulary = set.union(*(ngrams(task()[0]) for task in TASKMAP['cross-validation'].values()))
  for task in TASKMAP['train-test split'].values():
    for partition in ['train', 'test']:
      try:
        vocabulary = vocabulary.union(ngrams(task(partition)[0]))
      except FileNotFoundError:
        pass
  return sorted(vocabulary)


def ntokens(tokens):
  return len([split_on_punctuation(' '.join(tokens))])


def synset_context(sents):
  def context(strdoc, intdoc, vocabulary, wndo2=None, unkfeat=None):
    unk = not unkfeat is None
    wndo2 = len(intdoc) if wndo2 is None else wndo2
    offset = 0
    for chunk in next(sents):
      if type(chunk) == list:
        length = ntokens(chunk)
      else:
        label = chunk.label()
        if type(label) == str:
          length = ntokens(chunk)
        else:
          length = ntokens(chunk[0])
          synset = label.synset()
          if synset in vocabulary:
            yield synset, chain(intdoc[offset-wndo2:offset], intdoc[offset+length:offset+length+wndo2])
          elif unk:
            yield unkfeat, chain(intdoc[offset-wndo2:offset], intdoc[offset+length:offset+length+wndo2])
      offset += length
  return context


def synset_vocab():
  return sorted({label.synset() for label in (chunk.label() for chunk in semcor.tagged_chunks(tag='sem') if not type(chunk) == list) if not type(label) == str})


def alacache(nameroot, feature='ngram'):
  ''' function to return output of this script
  Args:
    nameroot: root of files (without extensions); the input argument 'outputroot'
    feature: string name of feature that was computed
  Returns:
    if file is for word x word cooccurrence: returns cooc matrix, word vocab, word counts; otherwise also returns feature vocab and featurecounts
  '''

  matrix = sp.load_npz(nameroot+'.npz')
  with open(nameroot+'.pkl', 'rb') as f:
    data = pickle.load(f)
  if len(data) == 2:
    return matrix, data['words'], data['counts']
  return matrix, data['words'], data['wordcounts'], data[feature+'s'], data[feature+'counts']


if __name__ == '__main__':

  feature = sys.argv[1]
  if feature == 'ngram':
    corpusfile, vocabfile, outputroot, n = sys.argv[2:6]
    n = int(n)
  elif feature == 'synset':
    vocabfile, outputroot = sys.argv[2:4]
  elif feature == 'word':
    feature = ''
    corpusfile, vocabfile, outputroot = sys.argv[2:5]
  else:
    raise(NotImplementedError)

  with open(vocabfile, 'r') as f:
    vocab = [line.split(' ')[0] for line in f]
    vocab.append(UNK)

  try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
  except ImportError:
    comm = None

  if feature:

    if feature == 'ngram':
      featurevocab = ngram_vocab(n)
      with open(corpusfile, 'r') as f:
        matrix, featurecounts, wordcounts = cooc_matrix((line.split() for line in f), featurevocab, vocab, n=n, unk=UNK, verbose=True, comm=comm)
    elif feature == 'synset':
      featurevocab = synset_vocab()
      matrix, featurecounts, wordcounts = cooc_matrix(semcor.sents(), featurevocab, vocab, doc2wnd=synset_context(iter(semcor.tagged_sents(tag='sem'))), unk=UNK, interval=100, verbose=True, wndo2=None)
      featurevocab = [synset.name() for synset in featurevocab]
    else:
      raise(NotImplementedError)

    if comm is None or not comm.rank:
      sp.save_npz(outputroot+'.npz', matrix)
      with open(outputroot+'.pkl', 'wb') as f:
        pickle.dump({'words': vocab, feature+'s': featurevocab, 'wordcounts': wordcounts, feature+'counts': featurecounts}, f)
    else:
      sys.exit()

  else:

    with open(corpusfile, 'r') as f:
      matrix, counts = symmetric_cooc_matrix((line.split() for line in f), vocab, unk=UNK, verbose=True, comm=comm)
    if comm is None or not comm.rank:
      sp.save_npz(outputroot+'.npz', matrix)
      with open(outputroot+'.pkl', 'wb') as f:
        pickle.dump({'words': vocab, 'counts': counts}, f)
    else:
      sys.exit()
