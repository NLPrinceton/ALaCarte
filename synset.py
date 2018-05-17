import csv
import pdb
import pickle
import sys
from collections import defaultdict
from operator import itemgetter
import numpy as np
np.seterr(all='raise')
from bs4 import BeautifulSoup as BS
from nltk.corpus import wordnet as wn
from numpy.linalg import norm
from scipy import sparse as sp
from ALaCarte.compute import *
from ALaCarte.cooc import *
from text_embedding.features import *
from text_embedding.vectors import *


BROWNROOT = '/n/fs/nlpdatasets/ALaCache/Brown' # output of running ALaCarte/cooc.py on Brown Corpus (can get corpus from http://nlp.cs.princeton.edu/ALaCarte/corpora)
SYNSETROOT = '/n/fs/nlpdatasets/ALaCache/Brown_Synset' # output of running ALaCarte/cooc.py on synset features using Bronw Corpus vocabulary (can get vocab from http://nlp.cs.princeton.edu/ALaCarte/corpora)
FILEDIR = os.path.dirname(os.path.realpath(__file__)) + '/'
POSMAP = {'j': 'as', 'n': 'n', 'r': 'r', 'v': 'v'}
DIM = 300
VECTORFILE = '/n/fs/nlpdatasets/NLPrinceton/enwiki-20161001_glove_300.txt' # GloVe trained on Wikipedia (can get vectors from http://nlp.cs.princeton.edu/ALaCarte/vectors)


def SemEval2013Task12():
  with open(FILEDIR+'data-SemEval2013_Task12/test/keys/gold/wordnet/wordnet.en.key', 'r') as f:
    gold = [(split[1], set(key.split('%')[1] for key in split[2:])) for split in (line.split() for line in f)]
  ids = {entry[0] for entry in gold}
  with open(FILEDIR+'data-SemEval2013_Task12/test/data/multilingual-all-words.en.xml', 'r') as f:
    soup = BS(f, 'lxml')
  data = [(inst['id'], inst['lemma'], inst['pos'], list(split_on_punctuation(' '.join(child.text for child in sent.children if not child == inst and not child == '\n').lower().replace('_', ' ')))) for text in soup('text') for sent in text('sentence') for inst in sent('instance') if inst['id'] in ids]
  return data, gold


def SemEval2015Task13():
  with open(FILEDIR+'data-SemEval2015_Task13/test/keys/gold_keys/EN/semeval-2015-task-13-en.key', 'r') as f:
    gold = [(split[1], set(key.split('%')[1] for key in split[2:] if key[:3] == 'wn:')) for split in (line.split() for line in f if '\twn:' in line)]
  ids = {entry[0] for entry in gold}
  with open(FILEDIR+'data-SemEval2015_Task13/test/data/semeval-2015-task-13-en.xml','r') as f:
    soup = BS(f, 'lxml')
  data = [(wf['id'], wf['lemma'], wf['pos'], list(split_on_punctuation(' '.join(child.text for child in sent.children if not child == wf and not child == '\n')))) for text in soup('text') for sent in text('sentence') for wf in sent('wf') if wf['id'] in ids and wf['pos'][0].lower() in POSMAP and wn.synsets(wf['lemma'], POSMAP[wf['pos'][0].lower()])]
  id2keys = defaultdict(lambda: set())
  for entry, keys in gold:
    id2keys[entry] = id2keys[entry].union(keys)
  gold = [(entry, id2keys[entry]) for entry, _, _, _ in data]
  return data, gold


def evaluate(retrieved, truth):
  precision = np.array([r in t[1] for r, t in zip(retrieved, truth)])
  recall = np.array([(r in t[1])/len(t[1]) for r, t in zip(retrieved, truth)])
  return np.mean(precision), np.mean(recall), 2.0*sum((precision*recall)[precision]/(precision+recall)[precision])/len(truth)


def cossim(u, v):
  normu = norm(u)
  if normu:
    normv = norm(v)
    if normv:
      return np.inner(u, v)/normu/normv
  return 0.0


def wordnet():

  write('Training Context Transform\n')
  cooc, words, counts = alacache(BROWNROOT)
  wordset = set(words)
  select = np.array([word in wordset for word in words])
  C = cooc[select][:,select]
  words = [word for sel, word in zip(select, words) if sel]
  counts = counts[select]
  X = vocab2mat(words, vectorfile=VECTORFILE, dimension=DIM, unit=False)
  A = linear_transform(C, X, counts, weights=np.log(counts), fit_intercept=False, n_jobs=-1)

  write('Constructing Synset Embeddings\n')
  cooc, _, _, synsets, synsetcounts = alacache(SYNSETROOT, 'synset')
  s2v = dict(zip(synsets, A.predict(cooc[:,select].dot(X)/synsetcounts[:,None])))

  output = {'counts': dict(zip(synsets, synsetcounts))}

  for task, load in [('SemEval2013 Task 12', SemEval2013Task12), ('SemEval2015 Task 13', SemEval2015Task13)]:
    alldata, allgold = load()
    write('Evaluating WSD on '+task+'\n')
    mfs = []
    alc = []
    gls = []
    cmb = []
    truth = []
    output[task] = {}
    for pos in sorted(POSMAP.values()):
      try:
        data, gold = zip(*((dentry, gentry) for dentry, gentry in zip(alldata, allgold) if POSMAP[dentry[2][0].lower()] == pos))
        output[task][pos] = {}
      except ValueError:
        continue
      write('\tPOS: '+pos+'\n')
      truth.extend(gold)

      s2c = {synset: count for synset, count in zip(synsets, synsetcounts)}
      keys = [max(((synset.lemmas()[0].key().split('%')[1], s2c.get(synset.name(), 0)) for synset in wn.synsets(entry[1], POSMAP[entry[2][0].lower()])), key=itemgetter(1))[0] for entry in data]
      pr, re, f1 = evaluate(keys, gold)
      write('\t\tMF Sense  : P='+str(pr)+', R='+str(re)+', F1='+str(f1)+'\n')
      mfs.extend(keys)

      w2v = vocab2vecs(wordset.union({word for entry in data for word in entry[-1]}), vectorfile=VECTORFILE, dimension=DIM, unit=False)
      z = np.zeros(DIM)
      convecs = [A.coef_.dot(sum((w2v.get(word, z) for word in entry[-1]), z)) for entry in data]
      keys = [max(((synset.lemmas()[0].key().split('%')[1], cossim(s2v.get(synset.name(), z), convec)) for synset in wn.synsets(entry[1], POSMAP[entry[2][0].lower()])), key=itemgetter(1))[0] for entry, convec in zip(data, convecs)]
      pr, re, f1 = evaluate(keys, gold)
      write('\t\tA La Carte: P='+str(pr)+', R='+str(re)+', F1='+str(f1)+'\n')
      alc.extend(keys)
      for d, g, k in zip(data, gold, keys):
        correct = int(k in g[1])
        for synset in wn.synsets(d[1], POSMAP[d[2][0].lower()]):
          if synset.lemmas()[0].key().split('%')[1] in g[1]:
            name = synset.name()
            output[task][pos].setdefault(name, [0, 0])
            output[task][pos][name][correct] += 1

      tasksyns = {synset for entry in data for synset in wn.synsets(entry[1], POSMAP[entry[2][0].lower()])}

      w2v = vocab2vecs({word for synset in tasksyns for sent in [synset.definition()]+synset.examples() for word in split_on_punctuation(sent.lower())}, vectorfile=VECTORFILE, dimension=DIM, unit=False)
      glsvecs = {}
      for synset in tasksyns:
        glsvecs[synset] = np.zeros(DIM)
        lemmas = set.union(*(set(lemma.name().lower().split('_')) for lemma in synset.lemmas()))
        for sent in [synset.definition()]+synset.examples():
          for word in split_on_punctuation(sent.lower()):
            if not word in lemmas:
              glsvecs[synset] += w2v.get(word, z)
        glsvecs[synset] = A.coef_.dot(glsvecs[synset])
      keys = [max(((synset.lemmas()[0].key().split('%')[1], cossim(glsvecs.get(synset, z), convec)) for synset in wn.synsets(entry[1], POSMAP[entry[2][0].lower()])), key=itemgetter(1))[0] for entry, convec in zip(data, convecs)]
      pr, re, f1 = evaluate(keys, gold)
      write('\t\tGloss-Only: P='+str(pr)+', R='+str(re)+', F1='+str(f1)+'\n')
      gls.extend(keys)

      keys = [max(((synset.lemmas()[0].key().split('%')[1], cossim(s2v.get(synset.name(), glsvecs.get(synset, z)), convec)) for synset in wn.synsets(entry[1], POSMAP[entry[2][0].lower()])), key=itemgetter(1))[0] for entry, convec in zip(data, convecs)]
      pr, re, f1 = evaluate(keys, gold)
      write('\t\tCombined  : P='+str(pr)+', R='+str(re)+', F1='+str(f1)+'\n')
      cmb.extend(keys)

    write('\tAll POS  \n')
    for meth, keys in [('MF Sense  ', mfs), ('A La Carte', alc), ('Gloss-Only', gls), ('Combined  ', cmb)]:
      pr, re, f1 = evaluate(keys, truth)
      write('\t\t'+meth+': P='+str(pr)+', R='+str(re)+', F1='+str(f1)+'\n')


if __name__ == '__main__':

  wordnet()
