# ALaCarte

This repository contains code and transforms to induce your own rare-word/n-gram vectors as well as evaluation code for the [A La Carte Embedding paper](http://aclweb.org/anthology/P18-1002). 
An overview is provided in this [blog post](http://www.offconvex.org/2018/09/18/alacarte/) at OffConvex.

If you find any of this code useful please cite the following:

    @inproceedings{khodak2018alacarte,
      title={A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors},
      author={Khodak, Mikhail and Saunshi, Nikunj and Liang, Yingyu and Ma, Tengyu and Stewart, Brandon and Arora, Sanjeev},
      booktitle={Proceedings of the ACL},
      year={2018}
    }

# Inducing your own à la carte vectors

The following are steps to induce your own vectors for rare words or n-grams in the same semantic space as [existing GloVe embeddings](https://nlp.stanford.edu/projects/glove/).
For rare words from the IMDB, PTB-WSJ, SST, and STS tasks you can find vectors induced using Common Crawl / Gigaword+Wikipedia at http://nlp.cs.princeton.edu/ALaCarte/vectors/induced/.

1. Make a text file containing one word or space-delimited n-gram per line. These are the targets for which vectors are to be induced.
2. Download source embedding files, which should have the format "word float ... float" on each line. Can find GloVe embeddings [here](https://nlp.stanford.edu/projects/glove/). Choose the appropriate transform in the <tt>transform</tt> directory.
3. If using Common Crawl, download a file of WET paths (e.g. [here](https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2014-52) for the 2014 crawl). Run <tt>alacarte.py</tt> with this passed to the <tt>--paths</tt> argument. Otherwise pass (one or more) text files to the <tt>--corpus</tt> argument. 

*Dependencies:*

Required: numpy

Optional: h5py (check-pointing), nltk (n-grams), cld2-cffi (checking English), mpi4py (parallelizing using MPI), boto (Common Crawl)

*For inducing vectors from Common Crawl on an AWS EC2 instance:*

1. Start an instance. Best to use a memory-optimized (<tt>r4.*</tt>) Linux instance.
2. Download and execute <tt>install.sh</tt>.
3. Upload your list of target words to the instance and run <tt>alacarte.py</tt>.

# Evaluation code

  * http://nlp.cs.princeton.edu/ALaCarte (GloVe Vectors)
  * http://nlp.cs.princeton.edu/CRW (CRW Dataset)
  
Note that the code in this directory treats adding up all embeddings of context words in a corpus as a matrix operation. This is memory-intensive and more practical implementations should use simple vector addition to compute context vectors.
  
Dependencies: nltk, numpy, scipy, text_embedding

Optional: mpi4py (to parallelize coocurrence matrix construction)
