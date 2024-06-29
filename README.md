# Protein Sequence embedding comparison for Sub-cellular localization using DeepLoc 2.0

Multi-label subcellular localization based on traditional and Machine Learning based embeddings.

This work has been inspired by
1. [DeepLoc 2.0: multi-label subcellular localization prediction using protein language
models](https://academic.oup.com/nar/article/50/W1/W228/6576357)
2. [Amino Acid Encoding Methods for Protein Sequences: A Comprehensive Review and 
Assessment](https://link.springer.com/article/10.1186/s12859-020-03546-x)
3. [Amino acid encoding for deep learning applications](https://ieeexplore.ieee.org/abstract/document/8692651)

Please get the required data_files by cloning our GitHub repo
[Neural-Dreamers-Deeploc 2.0](https://github.com/Neural-Dreamers/DeepLoc-2.0)

## Data
The 'data_files' folder contains the data for training
1. multisub_5_partitions_unique.csv: Annotated SwissProt Sequences, labels, and partitions for subcellular localization
2. multisub_ninesignals.pkl, sorting_signals.csv: Annotated SwissProt Sequences and sorting signal annotations
3. Processed FASTA files for generating embeddings

## Models
Models dubbed Fast (ESM1b), Accurate (ProtT5), OneHot, BLOSUM (BLOSUM62) and Fast2 (ESM2b) are used. `<MODEL-TYPE>` refers to one of these. 

## Setup

It is recommended to set up a conda environment using

`conda env create -f environment.yml`

## Training Workflow

Training is divided into two stages:

`python train_sl.py --model <MODEL-TYPE>`
1. Generate and store embeddings for faster training. Note: h5 files of ~30-40 GB are stored in "data_files/embeddings".
2. Train sub-cellular localization and interpretable attention.
3. Generate predictions and intermediate representations for sorting signal prediction.
4. Compute metrics on the SwissProt CV dataset.
