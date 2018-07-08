import argparse
import os
import sys
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from glob import glob
from gzip import GzipFile
from tempfile import TemporaryFile
from unicodedata import category
import numpy as np
np.seterr(all='raise')


GLOVEFILE = 'glove.840B.300d.txt'
WETPATHSFILE = 'wet.paths'
TIMEOUT = 180
ATTEMPTS = 20
SPACE = ' '
CATEGORIES = {'M', 'P', 'S'}
MINENGPER = 90.0
MAXTOKLEN = 1000
FLOAT = np.float32
INT = np.uint64


def write(msg, comm=None):
  '''writes to std out
  Args:
    msg: string
    comm: MPI Communicator (will not write if not root process)
  Returns:
    length of msg
  '''
  
  if comm is None or not comm.rank:
    sys.stdout.write(msg)
    sys.stdout.flush()
  return len(msg)


def ranksize(comm=None):
  '''returns rank and size of MPI Communicator
  Args:
    comm: MPI Communicator
  Returns:
    int, int
  '''

  if comm is None:
    return 0, 1
  return comm.rank, comm.size


def checkpoint(comm=None):
  '''waits until all processes have reached this point
  Args:
    comm: MPI Communicator
  '''

  if not comm is None:
    comm.allgather(0)


def is_punctuation(char):
  '''checks if unicode character is punctuation
  '''

  return category(char)[0] in CATEGORIES


def subtokenize(token, vocab):
  '''crude tokenization based on given vocabulary
  Args:
    token: str with no spaces
    vocab: set or dict with str keys
  Returns:
    subtoken generator, where subtoken is a substring of token contained in vocab or False
  '''

  if token in vocab:
    yield token

  elif len(token) == 1 or len(token) > MAXTOKLEN:
    yield False

  else:

    # determines where to split on punctuation based on whether results are in the vocabulary
    first = is_punctuation(token[0])
    for i, char in zip(range(1, len(token)), token[1:]):
      if first ^ is_punctuation(char):
        a0 = token[:i]
        b0 = token[i:]
        a1 = a0 + char
        b1 = b0[1:]
        recurse = False
        if a0 in vocab:
          if not b0 in vocab:
            if a1 in vocab and b1 in vocab:
              a0, b0 = a1, b1
            else:
              recurse = True
        elif a1 in vocab:
          a0 = a1
          recurse = not b1 in vocab
        else:
          a0 = False
          if not b0 in vocab:
            if b1 in vocab:
              b0 = b1
            else:
              recurse = True
        yield a0
        for substr in subtokenize(b0, vocab):
          yield substr
        break
    else:
      yield False


class ALaCarteReader:
  '''reads documents and updates context vectors
  '''

  def __init__(self, w2v, targets, wnd=10, checkpoint=None, interval=[0, float('inf')], comm=None):
    '''initializes context vector dict as self.c2v and counts as self.target_counts
    Args:
      w2v: {word: vector} dict of source word embeddings
      targets: iterable of targets to find context embeddings for
      wnd: context window size (uses this number of words on each side)
      checkpoint: path to HDF5 checkpoint file (both for recovery and dumping)
      interval: corpus start and stop positions
      comm: MPI Communicator
    '''

    self.w2v = w2v
    self.combined_vocab = self.w2v
    gramlens = {len(target.split()) for target in targets}
    assert len(gramlens) == 1, "all target n-grams must have the same n"
    self.n = gramlens.pop()
    if self.n > 1:
      self.targets = [tuple(target.split()) for target in targets]
      self.target_vocab = set(self.targets)
      self.combined_vocab = {word for target in targets for word in target.split()}.union(self.combined_vocab)
    else:
      self.targets = targets
      self.target_vocab = set(targets)
      self.combined_vocab = self.target_vocab.union(self.combined_vocab)
    self.target_counts = Counter()

    dimension = next(iter(self.w2v.values())).shape[0]
    self.dimension = dimension
    self.zero_vector = np.zeros(self.dimension, dtype=FLOAT)
    self.c2v = defaultdict(lambda: np.zeros(dimension, dtype=FLOAT))

    self.wnd = wnd
    self.learn = len(self.combined_vocab) == len(self.target_vocab) and self.n == 1

    self.datafile = checkpoint
    self.comm = comm
    self.rank, self.size = ranksize(comm)
    position = interval[0]

    if self.rank:
      self.vector_array = FLOAT(0.0)
      self.count_array = INT(0)

    elif checkpoint is None or not os.path.isfile(checkpoint):
      self.vector_array = np.zeros((len(self.targets), dimension), dtype=FLOAT)
      self.count_array = np.zeros(len(self.targets), dtype=INT)

    else:

      import h5py

      f = h5py.File(checkpoint, 'r')
      position = f.attrs['position']
      assert interval[0] <= position < interval[1], "checkpoint position must be inside corpus interval"
      self.vector_array = np.array(f['vectors'])
      self.count_array = np.array(f['counts'])

    self.position = comm.bcast(position, root=0) if self.size > 1 else position
    self.stop = interval[1]

  def reduce(self):
    '''reduces data to arrays at the root process
    '''

    comm, rank, size = self.comm, self.rank, self.size
    targets = self.targets

    c2v = self.c2v
    dimension = self.dimension
    vector_array = np.vstack(c2v.pop(target, np.zeros(dimension, dtype=FLOAT)) for target in targets)

    target_counts = self.target_counts
    count_array = np.array([target_counts.pop(target, 0) for target in targets], dtype=INT)

    if rank:
      comm.Reduce(vector_array, None, root=0)
      comm.Reduce(count_array, None, root=0)
    elif size > 1:
      comm.Reduce(self.vector_array+vector_array, self.vector_array, root=0)
      comm.Reduce(self.count_array+count_array, self.count_array, root=0)
    else:
      self.vector_array += vector_array
      self.count_array += count_array

  def checkpoint(self, position):
    '''dumps data to HDF5 checkpoint
    Args:
      position: reader position
    Returns:
      None
    '''

    datafile = self.datafile
    assert not datafile is None, "no checkpoint file specified"
    self.reduce()

    if not self.rank:

      import h5py

      f = h5py.File(datafile+'~tmp', 'w')
      f.attrs['position'] = position
      f.create_dataset('vectors', data=self.vector_array, dtype=FLOAT)
      f.create_dataset('counts', data=self.count_array, dtype=INT)
      f.close()
      if os.path.isfile(datafile):
        os.remove(datafile)
      os.rename(datafile+'~tmp', datafile)
    self.position = position

  def target_coverage(self):
    '''returns fraction of targets covered (as a string)
    Args:
      None
    Returns:
      str (empty on non-root processes)
    '''

    if self.rank:
      return ''
    return str(sum(self.count_array>0)) + '/' + str(len(self.targets))

  def read_ngrams(self, tokens):
    '''reads tokens and updates context vectors
    Args:
      tokens: list of strings
    Returns:
      None
    '''

    import nltk

    # gets location of target n-grams in document
    target_vocab = self.target_vocab
    n = self.n
    ngrams = list(filter(lambda entry: entry[1] in target_vocab, enumerate(nltk.ngrams(tokens, n))))

    if ngrams:

      # gets word embedding for each token
      w2v = self.w2v
      zero_vector = self.zero_vector
      wnd = self.wnd
      start = max(0, ngrams[0][0] - wnd)
      vectors = [None]*start + [w2v.get(token, zero_vector) if token else zero_vector for token in tokens[start:ngrams[-1][0]+n+wnd]]
      c2v = self.c2v
      target_counts = self.target_counts

      # computes context vector around each target n-gram
      for i, ngram in ngrams:
        c2v[ngram] += sum(vectors[max(0, i-wnd):i], zero_vector) + sum(vectors[i+n:i+n+wnd], zero_vector)
        target_counts[ngram] += 1


  def read_document(self, document):
    '''reads document and updates context vectors
    Args:
      document: str
    Returns:
      None
    '''

    # tokenizes document
    combined_vocab = self.combined_vocab
    tokens = [subtoken for token in document.split() for subtoken in subtokenize(token, combined_vocab)]
    if self.n > 1:
      return self.read_ngrams(tokens)

    # eliminates tokens not within the window of a target word
    T = len(tokens)
    wnd = self.wnd
    learn = self.learn
    if learn:
      check = bool
    else:
      target_vocab = self.target_vocab
      check = lambda token: token in target_vocab
    try:
      start = max(0, next(i for i, token in enumerate(tokens) if check(token)) - wnd)
    except StopIteration:
      return None
    stop = next(i for i, token in zip(reversed(range(T)), reversed(tokens)) if check(token)) + 1 + wnd
    tokens = tokens[start:stop]
    T = len(tokens)

    # gets word embedding for each token
    w2v = self.w2v
    zero_vector = self.zero_vector
    vectors = [w2v.get(token, zero_vector) if token else zero_vector for token in tokens]
    context_vector = sum(vectors[:wnd+1])
    c2v = self.c2v
    target_counts  = self.target_counts

    # slides window over document
    for i, (token, vector) in enumerate(zip(tokens, vectors)):
      if token and (learn or token in target_vocab):
        c2v[token] += context_vector - vector
        target_counts[token] += 1
      if i < T - 1:
        right_index = wnd + 1 + i
        if right_index < T and tokens[right_index]:
          context_vector += vectors[right_index]
        left_index = i - wnd
        if left_index > -1 and tokens[left_index]:
          context_vector -= vectors[left_index]


def is_english(document):
  '''checks if document is in English
  '''

  import cld2

  reliable, _, details = cld2.detect(document, bestEffort=True)
  return reliable and details[0][0] == 'ENGLISH' and details[0][2] >= MINENGPER 


def make_printable(string):
  '''returns printable version of given string
  '''

  return ''.join(filter(str.isprintable, string))


def process_documents(func):
  '''wraps document generator function to handle English-checking and lower-casing and to return data arrays
  '''

  def wrapper(string, reader, verbose=False, comm=None, english=False, lower=False):

    generator = (make_printable(document) for document in func(string, reader, verbose=verbose, comm=comm))
    if english:
      generator = (document for document in generator if is_english(document))
    if lower:
      generator = (document.lower() for document in generator)

    for i, document in enumerate(generator):
      reader.read_document(document)

    reader.reduce()
    write('\rFinished Processing Corpus; Targets Covered: '+reader.target_coverage()+' \n', comm)
    return reader.vector_array, reader.count_array

  return wrapper


@process_documents
def wet_documents(pathsfile, reader, verbose=False, comm=None):
  '''iterates over Common Crawl WET files
  Args:
    pathsfile: file with a Common Crawl filepath on each line
    reader: ALaCarteReader object
    verbose: display progress
    comm: MPI Communicator
  Returns:
    str generator distributing documents across processes
  '''

  import boto3
  import botocore

  position = reader.position
  if position == -1:
    return
  rank, size = ranksize(comm)
  client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED, read_timeout=TIMEOUT, retries={'max_attempts': ATTEMPTS}))
  with open(pathsfile, 'r') as f:
    paths = [line for line in f]

  for i, path in enumerate(paths):
    if i < position:
      continue

    if i > position and not i%1000:
      reader.reduce()
      if verbose and not rank:
        write('\rProcessed '+str(i)+'/'+str(len(paths))+' Paths; Target Coverage: '+reader.target_coverage(), comm)
      if not reader.datafile is None:
        reader.checkpoint(i)
    if i >= reader.stop:
      break

    if i%size == rank:
      temp = TemporaryFile('w+b')
      client.download_fileobj('commoncrawl', path.strip(), temp)
      temp.seek(0)
      for document in GzipFile(fileobj=temp).read().decode('utf-8').split('WARC/1.0')[2:]:
        try:
          yield document[document.index('\n', document.index('Content-Length')):].strip()
        except ValueError:
          pass


@process_documents
def corpus_documents(corpusfile, reader, verbose=False, comm=None):
  '''iterates of text document
  Args:
    corpusfile: text file with a document on each line
    reader: ALaCarteReader object
    verbose: display progress
    comm: MPI Communicator
  Returns:
    str generator distributing documents across processes
  '''

  position = reader.position
  rank, size = ranksize(comm)
  with open(corpusfile, 'r') as f:

    f.seek(position)
    line = f.readline()
    i = 0
    while line:

      if i and not i%1000000:
        reader.reduce()
        if verbose and not rank:
          write('\rProcessed '+str(i)+' Lines; Target Coverage: '+reader.target_coverage(), comm)
        if not reader.datafile is None:
          reader.checkpoint(f.tell())
      if i >= reader.stop:
        break

      if i%size == rank:
        yield line.strip()

      line = f.readline()
      i += 1


def load_vectors(vectorfile):
  '''loads word embeddings from .txt
  Args:
    vectorfile: .txt file in "word float ... " format
  Returns:
    (word, vector) generator
  '''

  words = set()
  with open(vectorfile, 'r') as f:
    for line in f:
      index = line.index(SPACE)
      word = make_printable(line[:index])
      if not word in words:
        words.add(word)
        yield word, np.fromstring(line[index+1:], dtype=FLOAT, sep=SPACE)


def dump_vectors(generator, vectorfile):
  '''dumps embeddings to .txt
  Args:
    generator: (gram, vector) generator; vector can also be a scalar
    vectorfile: .txt file
  Returns:
    None
  '''

  with open(vectorfile, 'w') as f:
    for gram, vector in generator:
      numstr = ' '.join(map(str, vector.tolist())) if vector.shape else str(vector)
      f.write(gram+' '+numstr+'\n')


def context_vectors(targets, w2v, documents, wnd=10, checkpoint=None, comm=None):
  '''constructs context vectors from documents
  Args:
    targets: ordered iterable of words to find context embeddings for
    w2v: {word: vector} dict of source word embeddings
    documents: function taking two optional function arguments that returns a document generator
    wnd: context window size
    checkpoint: path to HDF5 checkpoint file
    comm: MPI Communicator
  Returns:
    numpy array of size (len(targets), dimension), numpy array of size (len(targets),); at root process only
  '''
    
  rank, size = ranksize(comm)

  for i, document in enumerate(documents(alc)):
    alc.read_document(document)
    if not (i+1)%1000:
      write('\r'+str(i+1)+' Documents', comm)
  alc.reduce()
  return alc.vector_array, alc.count_array


def parse():
  '''parses command-line arguments
  '''

  parser = argparse.ArgumentParser(prog='python alacarte.py')

  parser.add_argument('dumproot', help='root of file names for intermediate and output dumps', type=str)
  parser.add_argument('-m', '--matrix', help='binary file for a la carte transform matrix', type=str)
  parser.add_argument('-v', '--verbose', action='store_true', help='display progess')
  parser.add_argument('-r', '--restart', help='HDF5 checkpoint file for restarting', type=str)
  parser.add_argument('-i', '--interval', nargs=2, default=['0', 'inf'], help='corpus position interval')

  parser.add_argument('-s', '--source', default=GLOVEFILE, help='source word embedding file', type=str)
  parser.add_argument('-p', '--paths', default=WETPATHSFILE, help='location of WET paths file', type=str)
  parser.add_argument('-c', '--corpus', nargs='*', help='list of text corpus files')
  parser.add_argument('-t', '--targets', help='target word file', type=str)

  parser.add_argument('-w', '--window', default=10, help='size of context window', type=int)
  parser.add_argument('-e', '--english', action='store_true', help='check documents for English')
  parser.add_argument('-l', '--lower', action='store_true', help='lower-case documents')

  return parser.parse_args()


def main(args, comm=None):
  '''a la carte embedding induction
  '''

  rank, size = ranksize(comm)
  root = args.dumproot
  matrixfile = root+'_transform.bin' if args.matrix is None else args.matrix

  write('Source Embeddings: '+args.source+'\n', comm)
  w2v = OrderedDict(load_vectors(args.source))

  if args.targets is None:
    write('No Targets Given; Will Learn Induction Matrix and Dump to '+matrixfile+'\n', comm)
    targets = w2v.keys()
    M = None
  else:
    write('Loading Targets from '+args.targets+'\n', comm)
    with open(args.targets, 'r') as f:
      targets = [target for target in (line.strip() for line in f) if not target in w2v]
    assert len(targets), "no uncovered targets found"
    assert len(set(len(target.split()) for target in targets)) == 1, "all target n-grams must have the same n"
    write('Induction Matrix: '+matrixfile+'\n', comm)
    assert os.path.isfile(matrixfile), "induction matrix must be given if targets given"
    M = np.fromfile(matrixfile, dtype=FLOAT)
    d = int(np.sqrt(M.shape[0]))
    assert d == next(iter(w2v.values())).shape[0], "induction matrix dimension and word embedding dimension must be the same"
    M = M.reshape(d, d)

  if args.restart:
    write('Checkpoint: '+args.restart+'\n', comm)
  interval = [int(args.interval[0]), int(args.interval[1])] if args.interval[1].isdigit() else [int(args.interval[0]), float(args.interval[1])]
  alc = ALaCarteReader(w2v, targets, wnd=args.window, checkpoint=args.restart, interval=interval, comm=comm)

  write('Building Context Vectors\n', comm)
  if args.corpus:
    context_vectors = FLOAT(0.0)
    target_counts = INT(0)
    for corpus in args.corpus:
      write('Source Corpus: '+corpus+'\n', comm)
      context_vectors, target_counts = corpus_documents(corpus, alc, verbose=args.verbose, comm=comm, english=args.english, lower=args.lower)
      if args.restart:
        alc.checkpoint(0)
  else:
    write('Source Corpus: WET Files in '+args.paths+'\n', comm)
    context_vectors, target_counts = wet_documents(args.paths, alc, verbose=args.verbose, comm=comm, english=args.english, lower=args.lower)
    if args.restart:
      alc.checkpoint(-1)

  if rank:
    sys.exit()

  nz = target_counts > 0

  if M is None:

    from sklearn.linear_model import LinearRegression as LR
    from sklearn.preprocessing import normalize

    write('Learning Induction Matrix\n', comm)
    X = np.true_divide(context_vectors[nz], target_counts[nz, None], dtype=FLOAT)
    Y = np.vstack(vector for vector, count in zip(w2v.values(), target_counts) if count)
    M = LR(fit_intercept=False).fit(X, Y).coef_.astype(FLOAT)
    write('Finished Learning Transform; Average Cosine Similarity: '+str(np.mean(np.sum(normalize(X.dot(M.T))*normalize(Y), axis=1)))+'\n', comm)

    write('Dumping Induction Transform to '+matrixfile+'\n', comm)
    dump_vectors(zip(targets, target_counts), root+'_source_vocab_counts.txt')
    context_vectors.tofile(root+'_source_context_vectors.bin')
    M.tofile(matrixfile)

  else:

    write('Dumping Induced Vectors to '+root+'_alacarte.txt\n', comm)
    dump_vectors(zip(targets, target_counts), root+'_target_vocab_counts.txt')
    context_vectors.tofile(root+'_target_context_vectors.bin')
    context_vectors[nz] = np.true_divide(context_vectors[nz], target_counts[nz, None], dtype=FLOAT)
    dump_vectors(zip(targets, context_vectors.dot(M.T)), root+'_alacarte.txt')


if __name__ == '__main__':

  try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
  except ImportError:
    comm = None

  main(parse(), comm=comm)
