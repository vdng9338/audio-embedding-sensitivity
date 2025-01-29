Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects
================

This repository contains the code (Jupyter notebooks and Python scripts) associated with the following paper:

> Victor Deng, Changhong Wang, Gaël Richard, Brian McFee. "Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects", in _International Conference on Acoustics, Speech, and Signal Processing (ICASSP)_, 2025.

The experiments, aligned with the section order in the paper, can be run by following the instructions outlined below.

# Requirements

To run the experiments, you will need Conda (e.g. Miniconda, see https://docs.anaconda.com/miniconda/install/ for instructions to install Miniconda).

All the software needed to run the experiments will be managed by Conda and will reside in Conda environments. Instructions to set up the Conda environments for each part of the experiments are provided in the relevant sections of this README.

We worked using Ubuntu 24.04, but the code should run on most popular Linux distributions.

# First steps

## Getting the code and setting up the directories

First of all, create a directory that will contain all the files related to the experiments. We will use `~/audio-embedding-sensitivity` as our path in the following; in all the instructions below, replace any occurrence of `~/audio-embedding-sensitivity` with your chosen path.

- Clone the repository containing the code into `~/audio-embedding-sensitivity/code-repo`:
```
~/audio-embedding-sensitivity$ git clone https://github.com/vdng9338/audio-embedding-sensitivity.git code-repo
```

## Getting the IRMAS training dataset

- Create a directory `~/audio-embedding-sensitivity/datasets` that will store the original and effected audio samples.
- We make use of the IRMAS training data at https://zenodo.org/records/1290750#.WzCwSRyxXMU . Download [IRMAS-TrainingData.zip](https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1) and extract its contents into `~/audio-embedding-sensitivity/datasets`:
```
~/audio-embedding-sensitivity$ cd datasets
~/audio-embedding-sensitivity/datasets$ unzip path/to/IRMAS-TrainingData.zip
# unzip output
~/audio-embedding-sensitivity/datasets$ cd ..
```

# Extracting embeddings and applying audio effects (Part II.A)

We do not provide precomputed embeddings of the IRMAS training dataset, because the resulting files occupy a lot of storage. We therefore invite readers to extract the embeddings themselves using the following instructions.

## Installing the dependencies for embedding extraction and application of audio effects

For the purpose of extracting embeddings, we create three Conda environments, one for each foundation model.

### OpenL3

Change directory to `~/audio-embedding-sensitivity/code-repo`. Then first run either:
```bash
conda env create -f environments/environment-openl3-extract.yml
conda activate openl3-extract
```
or (to try using more recent versions of the packages):
```bash
conda create -n openl3-extract python=3.8
conda activate openl3-extract
conda install -c conda-forge numpy=1.24.3 tqdm librosa h5py
pip install openl3 pedalboard

# For TensorFlow GPU support; commands from https://web.archive.org/web/20230926140206/https://www.tensorflow.org/install/pip
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
```
**and then** run:
```bash
# For TensorFlow GPU support; commands from https://web.archive.org/web/20230926140206/https://www.tensorflow.org/install/pip (continued)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### PANNs

Run the following command (still from `~/audio-embedding-sensitivity/code-repo`):
```bash
conda env create -f environments/environment-panns-extract.yml
```
Alternatively, you can run the following commands to try using more recent versions of the packages:
```bash
conda create -n panns-extract python=3.12 # at time of writing, Python 3.13 is not supported by PyTorch
conda activte panns-extract
conda install pip
conda install -c conda-forge h5py
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install panns-inference pedalboard
```

### CLAP

Run (still from `~/audio-embedding-sensitivity/code-repo`):
```bash
conda env create -f environments/environment-clap-extract.yml
```
Alternatively, you can run the following commands to try using more recent versions of the packages:
```bash
conda create -n clap-extract python=3.11
conda activate clap-extract
conda install -c conda-forge h5py
pip install msclap pedalboard
```

## Extracting the embeddings of the original audio files

In the file `~/audio-embedding-sensitivity/code-repo/2A.1_featureExtract_original.py` (referred to as `2A.1_featureExtract_original.py` in the rest of this section), change the definition of the `basedir` variable to the path of the directory containing all the experiment files (`~/audio-embedding-sensitivity`).

Then, to extract the embeddings, run:
```
~/audio-embedding-sensitivity/code-repo$ conda activate openl3-extract
(openl3-extract) ~/audio-embedding-sensitivity/code-repo$ python3 2A.1_featureExtract_original.py openl3
# (logging messages)
(openl3-extract) ~/audio-embedding-sensitivity/code-repo$ conda activate panns-extract
(panns-extract) ~/audio-embedding-sensitivity/code-repo$ python3 2A.1_featureExtract_original.py panns
# (logging messages)
# At that point, a reboot may be required in order not to run into an illegal memory access error
(panns-extract) ~/audio-embedding-sensitivity/code-repo$ conda activate clap-extract
(clap-extract) ~/audio-embedding-sensitivity/code-repo$ python3 2A.1_featureExtract_original.py clap
# (logging messages)
```

This will create or update the file `embeddings/embeddings.h5` in the `code-repo` folder.

By default, embedding extraction will use a GPU on your system. To perform all calculations on the CPU instead, replace (as instructed in the comments):
```python
elif embedding_name == "panns":
    model = AudioTagging(checkpoint_path=None, device="cuda") # change to "cpu" to use CPU
else:
    model = CLAP(version='2023', use_cuda=True) # change to False to use CPU
```
with:
```python
elif embedding_name == "panns":
    model = AudioTagging(checkpoint_path=None, device="cpu") # change to "cpu" to use CPU
else:
    model = CLAP(version='2023', use_cuda=False) # change to False to use CPU
```

## Applying audio effects and extracting embeddings of effected samples

First, change the value of the variable `basedir` to the base directory of the experiments:
```python
###########################################################
# Replace this with the base directory containing all the #
# files related to the experiments                        #
###########################################################
basedir = "/home/user/audio-embedding-sensitivity"
```

Then, to extract the effected embeddings for effect `$EFFECT` (bitcrush, lowpass_cheby, gain or reverb) and embedding `$EMBEDDING` (openl3, panns or clap), run the following commands (from the `code-repo` folder):
```
$ conda activate $EMBEDDING-extract
$ python3 2A.2_featureExtract_audioEffects.py $EFFECT $EMBEDDING -1 -1 # to get the number of parameters in the grid
There are XXX parameters in the grid
$ python3 2A.2_featureExtract_audioEffects.py $EFFECT $EMBEDDING 0 [XXX-1] # without the square brackets; to perform extraction for all parameter values successively
```
Extraction can be parallelized by running the script several times with disjoint parameter ranges. To extract the embeddings for parameter ranks `m` to `n`, with the $EMBEDDING-extract Conda environment activated, run (without the square brackets):
```
$ python3 2A.2_featureExtract_audioEffects.py $EFFECT $EMBEDDING [m] [n]
```

These commands will create (or update) the files `embeddings/embeddings_$EFFECT_[parameter value].h5`.

NOTE: Extraction of OpenL3 embeddings takes much longer than PANNs and CLAP embeddings. With a NVIDIA T4 GPU, extraction of OpenL3 embeddings for one effect and effect parameter takes around 25-30 min., vs. around 3-4 min. for PANNs and CLAP.


### (Optional) Recreate train-test split

The train-test split is stored in the file `train_test_split.csv`, where the columns indicate (in this order) the instrument, genre, filename and split (train/test) of each sample. You can either use the train-test split that we generated (CSV file provided in this repository), or regenerate it by running `python3 2A.3_train_test_split.py` (this needs `embeddings/embeddings.h5` to have been generated).

# Setting up the Conda environment for the experiments proper

The Conda environment for the experiments proper is described in `environments/environment.yml`. To create a Conda environment from it, run:
```bash
conda env create -f environments/environment.yml
```

Alternatively, to try to use the latest versions of the libraries, run:
```bash
conda create -n audio-emb-sensitivity
conda activate audio-emb-sensitivity
conda install -c conda-forge numpy tqdm pandas h5py scikit-learn scikit-learn-extra matplotlib ipympl notebook nb_conda_kernels jupyter ipywidgets umap-learn seaborn librosa
```

From now on, we will assume that you have activated the resulting environment (`audio-emb-sensitivity` by default).

# UMAP embeddings of isolated samples (Part II.B)

Plots are specific to a combination of embedding, audio effect and instrument.

As a preparation step, we average the embeddings per sample name (i.e. embeddings of the audio chunks corresponding to any given audio file are averaged together -- this only performs a nontrivial computation with OpenL3), group the embeddings corresponding to the different strengths of the chosen audio effect together and save these embeddings and the uneffected averaged embeddings corresponding to the given audio embedding and audio effect to an intermediate file (`embeddings/grouped/[embedding]/[audio effect].h5`). To do so, run:

```
python3 group_effect_params.py -e [effect] -b [embedding]
```
For help on the command, including a list of valid parameters for -e and -b, run `python3 group_effect_params.py -h`.

Then, open the Jupyter notebook `2B.1_umap_trajectories.ipynb` to plot the trajectories. This is an interactive notebook, whose functioning we hope is self-explanatory.

# CCA-related experiments (Part III)

## Computing CCA directions and correlation coefficients, and correlation plots

### Global CCA

Computation of the global CCA directions (which are still specific to a combination of embedding, audio effect and instrument) is performed as a part of the desensitization process. To perform the computation for a given combination of audio embedding and audio effect (but all instruments), run:

```
python3 desensitize_classif.py -b <embedding> -e <effect> -m cca -k 2
```

Computation of the (global) correlation coefficient and plotting of the correlation plots is done in the Jupyter notebook `3.1_global_cca_correlation.ipynb`. This notebook is interactive, except for instrument selection which needs to be done by changing the last line of the second cell of the notebook.

**TODO Test for more than one embedding-effect combination, check consistency with paper**

### Sample-wise CCA

- As a preprocessing step, we precompute the per-sample averaged embeddings of the input files. To do so for one combination of audio embedding, audio effect and effect parameter, run:
```
python3 average_embeddings.py train_test_split.csv embeddings/embeddings_<effect>_<parameter>.h5 <embedding> embeddings/averaged/<embedding>/embeddings_<effect>_<parameter>.h5
```
Typically, one wants to run this command for at least all the parameters of one given combination of audio effect and embedding.  
We provide a utility script, `average_all_embeddings.sh`, to run this command for all possible combinations. It should be run in the `audio-emb-sensitivity` Conda environment.
- To compute the sample-wise CCA directions, run:
```
python3 3.2_extract_samplewise_cca_dirs.py <embedding> <audio effect>
```
These are stored in `embeddings/averaged/<embedding>/ccadirs_<effect>.h5`.
- Computation of the sample-wise correlation coefficients, as well as plotting of the sample-wise correlation plots, is done in the notebook `3.3_samplewise_cca_correlation.ipynb`, which is interactive (but the last cell, which computes the correlation coefficients, needs to be re-run each time the audio effect or the embedding is changed).
- The distribution of the sample-wise R2 coefficients (correlation coefficients) is plotted in the notebook `3.4_samplewise_cca_r2_distrib.ipynb`.

## SVD of sample-wise CCA directions

The code for this experiment is included in the notebook `3.5_samplewise_cca_dirs_svd.ipynb`. This notebook requires the following commands to have been run (no need to run them again if you have run them in the previous subsection):
```
for emb in openl3 panns clap; do
    python3 average_embeddings.py train_test_split.csv embeddings/embeddings.h5 $emb embeddings/averaged/$emb/embeddings.h5
done
bash average_all_embeddings.sh
for emb in openl3 panns clap; do
    for eff in bitcrush gain lowpass_cheby reverb; do
        python3 3.2_extract_samplewise_cca_dirs.py $emb $eff
    done
done
```

# Attempting to reduce sensitivity of audio embeddings to audio effects (Part IV)

## Performing classification and computing performance

To perform classification and compute classification performance for a given combination of audio embedding, audio effect and list of desensitization methods, run:
```
python3 desensitize_classif.py -b <embedding> -e <effect> -m <comma-separated list of methods> -t <samplewise CCA SVD singular value relative threshold> -k 2
```
This will save the classification performance numbers in the folder `results/`.

ROC AUC plots are plotted in the notebook `4_classification_performance.ipynb`.
