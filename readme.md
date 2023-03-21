# Space-and-Speaker-Aware Acoustic Modeling with Effective Data Augmentation for Recognition of Multi-Array Conversational Speech

<p align="center">
  <a href="https://github.com/coalboss/SSA_AM/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/misitebao/standard-repository?style=flat-square"/></a>
  <a href="https://github.com/coalboss/SSA_AM"><img alt="GitHub Repo stars"src="https://img.shields.io/github/stars/coalboss/SSA_AM?style=flat-square"/></a>
  <a href="https://github.com/coalboss"><img alt="GitHub Repo stars" src="https://img.shields.io/badge/author-coalboss-brightgreen?style=flat-square"/></a>
</p>

<span id="nav-1"></span>

This repository provides a official implementation of our champion system of Track 1 of the CHiME-6 Challenge. 

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@inproceedings{chai2023ssa,
  title={Li Chai and Hang Chen and Jun Du and Qing-Feng Liu and Chin-Hui Lee},
  booktitle={Space-and-Speaker-Aware Acoustic Modeling with Effective Data Augmentation for Recognition of Multi-Array Conversational Speech},
  pages={},
  year={},
  organization={}
}
```

## Introduction

  The acoustic model of the ASR system is built largely following the Kaldi [CHIME6](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1) recipes which mainly contain two stages: GMM-HMM state model and TDNN deep learning model.

  - **GMM-HMM**

    For features extraction, we extract 13-dimensional MFCC features plus 3-dimensional pitches. As a start point for triphone models, a monophone model is trained on a subset of 50k utterances.  Then a small triphone model and a larger triphone model are consecutively trained using delta features on a subset of 100k utterances and the whole dataset respectively. In the third triphone model training process, an MLLT-based global transform is estimated iteratively on the top of LDA feature to extract independent speaker features. For the fourth triphone model, feature space maximum likelihood linear regression (fMLLR) with speaker adaptive training (SAT) is applied in the training.

  - **NN-HMM**

    Based on the tied-triphone state alignments from GMM, TDNN is configured and trained to replace GMM. Here two data augmentation technologies, speed-perturbation and volume-perturbation are applied on signal level. The input features are 40-dimensional high-resolution MFCC features with cepstral normalization. Note that for each frame-wise input, a 100-dimensional i-vector is also attached, whose extractor was trained on the expanded corpus. An advanced time-delayed neural network (TDNN) baseline using lattice-free maximum mutual information (LF-MMI) training and other strategies is adopted in the system, and you can consult the [paper](https://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) and the [document](https://kaldi-asr.org/doc/chain.html) for more details.

- ## Data Augmentation

- ## Acoustic Model

## Results

## Requirments

### Please star it, thank you! :）
