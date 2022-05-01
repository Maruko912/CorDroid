# Obfuscation-resilient Android Malware Analysis Based on Complementary Features

## Main requirements

  * **torch==1.7.0+cu101**
  * **efficientnet-pytorch==0.7.0**
  * **numpy==1.19.4**
  * **androguard** https://github.com/androguard/androguard

## Experiment

1. train fun2vec_model

   see feature_extract/fcg 

2. feature extraction

   Use the feature_extract script to extract the function call graph (including adjacency matrix, node feature matrix) and OMM features.

3. train OMM model and E-SFCG model

   use the omm_train/train script  to train OMM model, and use the sfcg_train/train script  to train E-SFCG model

4. Test Cordroid

   use test.py to test Cordroid

