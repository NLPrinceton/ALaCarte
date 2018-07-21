#!/bin/sh

disp() { echo -e "\e[44m\e[97m$@\e[49m\e[39m" ; }

disp "Installing GCC-C++ and OpenMPI"
sudo yum install gcc-c++ openmpi -y
echo 'export PATH="/usr/lib64/openmpi/bin:$PATH"' >> $HOME/.bashrc

disp "Downloading and Installing Miniconda"
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc

source ~/.bashrc
disp "Installing NumPy, MPI4Py, Scikit-Learn, Boto3, H5Py, CLD2, and NLTK"
conda install -y numpy mpi4py scikit-learn boto3 h5py nltk
pip install cld2-cffi

VERSION=840B.300d
#VERSION=42B.300d
disp "Downloading GloVe "$VERSION" Embeddings"
wget http://nlp.stanford.edu/data/glove.$VERSION.zip
unzip glove.$VERSION.zip

CRAWL=2014-52
disp "Downloading WET Paths for "$CRAWL" Crawl"
wget https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-$CRAWL/wet.paths.gz
gunzip wet.paths.gz

GIT=https://raw.githubusercontent.com/NLPrinceton/ALaCarte/master
disp "Downloading A La Carte Files"
wget $GIT/alacarte.py
wget $GIT/transform/$VERSION.bin
