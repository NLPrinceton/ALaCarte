import numpy as np
from gensim.models import Word2Vec
from numpy.linalg import norm
from scipy.stats import spearmanr
from text_embedding.documents import *
from ALaCarte.compute import *
from ALaCarte.cooc import *


COOCROOT = '/n/fs/nlpdatasets/ALaCache/Wiki' # output of running ALaCarte/cooc.py on Wikipedia (can get corpus from http://nlp.cs.princeton.edu/ALaCarte/corpora/)
FILEDIR = os.path.dirname(os.path.realpath(__file__)) + '/'
MODELFILE = '/n/fs/nlpdatasets/Nonce/wiki_all.model/wiki_all.sent.split.model' # obtain from http://clic.cimec.unitn.it/~aurelie.herbelot/wiki_all.model.tar.gz
MINCOUNT = 999


def load_nonces(partition):
  with open(FILEDIR+'data-nonces/n2v.definitional.dataset.'+partition+'.txt', 'r') as f:
    return zip(*((n, [w for w in d.split() if not w == '___']) for n, d in (line.split('\t') for line in f if not line[0] in {'#', '\n'})))


def rank_nonces(w2v, nonces, vectors):
  ranks = []
  SRR = 0.0
  for nonce, vector in zip(nonces, vectors):
    vector /= norm(vector)
    sim = np.inner(w2v[nonce], vector)
    r = sum(np.inner(v, vector)>sim for v in w2v.values())+1
    SRR += 1.0/r
    ranks.append(r)
  med = np.median(ranks)
  write('\rMedian='+str(med)+'; MRR='+str(SRR/len(nonces))+'\n')


def nonces(model, w2v, C, X, words, counts):

  write('Computing Transform\n')
  ntest, dtest = load_nonces('test')
  nset = set(ntest)
  select = np.array([not word in nset for word in words])
  counts = counts[select]
  A = linear_transform(C[select][:,select], X[select], counts, weights=counts>MINCOUNT, fit_intercept=False, n_jobs=-1).coef_
  z = np.zeros(A.shape[0])

  write('Evaluating on Nonce Task\n') 
  rank_nonces({w: v/norm(v) for w, v in w2v.items()}, ntest, (A.dot(sum((w2v.get(word, z) for word in defn), z)) for defn in dtest))


def load_chimeras(n, partition):
  with open(FILEDIR+'data-chimeras/dataset.l'+str(n)+'.tokenised.'+partition+'.txt', 'r', encoding='latin-1') as f:
    return zip(*(([[w for w in s.split() if not w == '___'] for s in fields[1].split('@@')], fields[2].split(','), [float(r) for r in fields[3].split(',')]) for fields in (line.strip().split('\t') for line in f)))


def eval_chimeras(w2v, probelists, ratlists, vectors):
  rhos = []
  for probes, ratings, vector in zip(probelists, ratlists, vectors):
    sims = []
    vector /= norm(vector)
    for i, probe in enumerate(probes):
      try:
        sims.append(np.inner(w2v[probe]/norm(w2v[probe]), vector))
      except KeyError:
        ratings = ratings[:i]+ratings[i+1:]
    rhos.append(spearmanr(ratings, sims))
  write('\ravg rho='+str(np.mean(rhos))+'\n')


def chimeras(model, w2v, C, X, counts):

  write('Computing Transform\n')
  A = linear_transform(C, X, counts, weights=counts>MINCOUNT, fit_intercept=False, n_jobs=-1).coef_
  z = np.zeros(model.vector_size)
  for n in [2, 4, 6]:
    write('Evaluating on Chimera-'+str(n)+' Task\n')
    sentlists, probelists, ratlists = load_chimeras(n, 'test')
    eval_chimeras(w2v, probelists, ratlists, (A.dot(sum((sum((w2v.get(word, z) for word in sent), z) for sent in sents), z)) for sents in sentlists))


if __name__ == '__main__':

  model = Word2Vec.load(MODELFILE)
  w2v = {word: model.wv[word] for word in model.wv.vocab.keys()}
  cooc, words, counts = alacache(COOCROOT)
  wordset = set(model.wv.vocab.keys())
  select = np.array([word in wordset for word in words])
  C = cooc[select][:,select]
  words = [word for sel, word in zip(select, words) if sel]
  counts = counts[select]
  X = np.vstack(model.wv[word] for word in words)

  nonces(model, w2v, C, X, words, counts)
  chimeras(model, w2v, C, X, counts)
