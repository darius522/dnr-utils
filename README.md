# Divide and Remaster Utility Scripts
Utility tools for the "Divide and Remaster" dataset, introduced as part of the Cocktail Fork problem paper

The DnR dataset is build from three, well-established, audio datasets; [Librispeech](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7178964), [Free Music Archive (FMA)](https://arxiv.org/pdf/1612.01840.pdf), and [Freesound Dataset 50k (FSD50K)](https://arxiv.org/pdf/2010.00475.pdf). We offer our dataset in both 16kHz and 44.1kHz sampling-rate along time-stamped annotations for each of the classes (genre for 'music', audio-tags for 'sound-effects', and transcription for 'speech'). We provide below more informations on how the dataset is build and what it's consists of exactly. We also go over the process of building the dataset from scratch for the cases it needs to.

-----

*   [Dataset Overview](#overview)
*   [Get the DnR Dataset](#getdnr)
    *   [Download](#download)
    *   [Building DnR From Scratch](#scratch)
*   [Resources and Support](#contact)

-----

<h2 id="overview">Dataset Overview</h2>
The Divide and Remix (DnR) dataset is a dataset aiming at providing research support for a relatively unexplored case of source separation with mixtures involving music, speech, and sound-effects (SFX) as their sources. The dataset is build from three, well-established, datasets. Consequently if one wants to build DnR from scratch, the aforementioned datasets will have to be downloaded first. Alternatively, DnR is also available on Zenodo

-----

<h2 id="getdnr">Get the DnR Dataset</h2>
In order to obtain DnR, several options are available depending on the task at hand:
<h3 id="download">Download</h3>

- DnR-HQ (44.1kHz) is available on Zenodo at the [following](#dummy) or simply run:
```
wget link_to_zenodo_dnr
```

- Alternatively, if DnR-16kHz is needed, please first download or build DnR-HQ locally. You can then downsample the dataset (either in-place or not) by running:
```
python dnr_utils.py --task=downsample --inplace=True
```

<h3 id="scratch">Building DnR From Scratch</h3>
In the section, we go over the DnR building process. Since DnR is directly drawn from *FSD50K*, *LibriSpeech*/*LibriVox*, and *FMA, we first need to download these datasets. Please head to the following links for more details on how to get them:

<h4 id="scratch">Datasets Downloads</h4>

<p align="center">

:police_car: <a href="https://zenodo.org/record/4060432#.YTkaoN8pBPY">FSD50K</a> :police_car:
<br>
:notes: <a href="https://github.com/mdeff/fma">FMA-Medium Set</a> :notes:
<br>
:speaking_head: <a href="https://www.openslr.org/12">LibriSpeech/LibriVox</a> :speaking_head:

<br><br>
</p>

Please note that for *FMA*, the [medium set](https://os.unil.cloud.switch.ch/fma/fma_medium.zip) only is required. In addition to the audio files, [the metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) should also be downloaded. For *LibriSpeech* DnR uses [dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz), [test-clean](https://www.openslr.org/resources/12/test-clean.tar.gz), and [train-clean-100](https://www.openslr.org/resources/12/train-clean-100.tar.gz). DnR will use the folder structure as well as metadata from LibriSpeech, but ultimately will build the LibriSpeech-HQ dataset off [the original LibriVox mp3s](https://www.openslr.org/resources/12/original-mp3.tar.gz), which is why we need them both for building DnR.

After download, all four datasets are expected to be found in the same root directory. Our root tree may look something like that. As the standardization script will look for specific file name, __please make sure that all directory names conform to the ones described below__:
```
root
├── fma-medium
│   ├── fma_metadata
│   │   ├── genres.csv
│   │   └── tracks.csv
│   ├── 008
│   ├── 008
│   ├── 009
│   └── 010
│   └── ...
├── fsd50k
│   ├── FSD50K.dev_audio
│   ├── FSD50K.eval_audio
│   └── FSD50K.ground_truth
│   │   ├── dev.csv
│   │   ├── eval.csv
│   │   └── vocabulary.csv
├── librispeech
│   ├── dev-clean
│   ├── test-clean
│   └── train-clean-100
└── librivox
    ├── 14
    ├── 16
    └── 17
    └── ...
```
-----

<h4 id="scratch">Datasets Standardization</h4>

Once all four datasets are downloaded, some standardization work needs to be taken care of. The standardization process can be be executed by running [`standardization.py`](https://github.com/darius522/dnr-utils/blob/d895d485bb6d2bde9d7a79faaa7a12172d45b375/standardization.py), which can be found in the [dnr-utils](https://github.com/darius522/dnr-utils) repository. Prior to running the script you may want to install all the necessary dependencies included as part of the `requirement.txt` with `pip install -r requirements.txt`. 
__Note:__ `pydub` uses `ffmpeg` under its hood, a system install of `fmmpeg` is thus required.
The standardization command may look something like:
```
python standardization.py --fsd50k-path=./FSD50K --fma-path=./FMA --librivox-path=./LibriVox --librispeech-path=./LibiSpeech  --dest-dir=./dest --validate-audio=True
```

<h4 id="scratch">DnR Dataset Compilation</h4>

Once the three resulting datasets are standardized, we are ready to finally compile DnR. At this point you should already have cloned the [dnr-utils](https://github.com/darius522/dnr-utils) repository, which contains two key files:

- [`config.py`](https://github.com/darius522/dnr-utils/blob/d895d485bb6d2bde9d7a79faaa7a12172d45b375/config.py) contains some configuration entries needed by the main script builder. You want to set all the appropriate paths pointing to your local datasets and ground truth files in there. 
- The compilation for a given set (here, `train`, `val`, and `eval`) can be executed with [`compile_dataset.py`](https://github.com/darius522/dnr-utils/blob/d895d485bb6d2bde9d7a79faaa7a12172d45b375/compile_dataset.py), for example by running the following commands __for each set__:
```
python compile_dataset.py with cfg.train
```
```
python compile_dataset.py with cfg.val
```
```
python compile_dataset.py with cfg.eval
```

-----

<h2 id="contact">Contact and Support</h2>

