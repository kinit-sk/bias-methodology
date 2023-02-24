# In-Depth Look at Word Filling Societal Bias Measures

This repository contains data and code for the _In-Depth Look at Word Filling Societal Bias Measures_ paper accepted to EACL 2023.

## Code

- `dataset.py` - Code for loading all the datasets that we use.
- `score.py` - Implementation of various bias measurments.
- `tokenization.py` - Functions related to tokenization and masking.
- `utils.py` - General utils.

## Notebooks

- `_crows.ipynb` - Visualization of the CrowS score for individual tokens.
- `_figures.ipynb` - Code that was used to generate figures and tables in the paper.
- `_score_visualization.ipynb` - Visualization of the score distribution for different models, scores and datasets.
- `_slovak_gender.ipynb` - Data analysis of the Slovak gender dataset created by us.
- `_stereoset.ipynb` - Data analysis of the StereoSet dataset.
- `_tokenization.ipynb` - Debug script used to analyze the tokenization scripts.

## Datasets

Following datasets can be found in `data` folder. Datasets with asterisk were used in our paper.

- `*crows_antistereotype.csv` - This is a 
- `*crows_negation.csv` - This is a 
- `crows_revised.csv` - The revised version of the CrowS-Pairs dataset. This is based on the version published by (Névéol 2022), but we have further revised it and fixed additional samples.
- `crows.csv` - The original CrowS-Pairs dataset. 
- `*slovak_gender.csv` - The Slovak gender dataset we collected.
- `*stereoset_genderswap_filtered.txt` - Our gender-swapped version of intersentence gender portion of StereoSet that was manually filtered to contain only gender bias samples.
- `*stereoset_genderswap.txt` - Our gender-swapped version of intrasentence gender portion of StereoSet.
- `stereoset_inter_genderswap.txt` - Our gender-swapped version of intersentence gender portion of StereoSet. This was not used in the paper.
- `*stereoset.json` - The original StereoSet dataset