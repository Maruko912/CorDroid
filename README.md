# Obfuscation-resilient Android Malware Analysis Based on Complementary Features

## Main requirements

  * **torch==1.7.0+cu101**
  * **efficientnet-pytorch==0.7.0**
  * **numpy==1.19.4**
  * **androguard** https://github.com/androguard/androguard

## Experiment

### 1. Feature Extraction
The following scripts can be used to extract E-SFCG and OMM features:
* feature_extract.py -- extract the E-SFCG (including adjacency matrix, node feature matrix) and OMM features.

### 2. Train OMM model and E-SFCG model
* omm_train/train.py -- train OMM model
* sfcg_train/train.py -- train E-SFCG model

### 3. Classification
* test.py -- fuse the two models to classify samples.

## Others
* OMM_heatmap/ --the OMM heatmap under the Reorder and Reflection obfuscation technologies.

