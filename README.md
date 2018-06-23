# ALaCarte

Evaluation code for A La Carte Embedding
  * https://arxiv.org/abs/1805.05388 (Manuscript)
  * http://nlp.cs.princeton.edu/ALaCarte (Data)
  * http://nlp.cs.princeton.edu/CRW (CRW Dataset)
  
Note that the code in this directory treats adding up all embeddings of context words in a corpus as a matrix operation. This is memory-intensive and more practical implementations should use simple vector addition to compute context vectors.
  
Dependencies: NLTK, NumPy, SciPy, text_embedding

Optional: mpi4py (to parallelize coocurrence matrix construction)

If you find this code useful please cite the following:

    @inproceedings{khodak2018alacarte,
      title={A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors},
      author={Khodak, Mikhail and Saunshi, Nikunj and Liang, Yingyu and Ma, Tengyu and Stewart, Brandon and Arora, Sanjeev},
      booktitle={To Appear in the Proceedings of the ACL},
      year={2018}
    }
