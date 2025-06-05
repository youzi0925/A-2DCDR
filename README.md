## A<sup>2</sup>DCDR
The source code is for the paper: **“Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features”**.  This implementation references the source code from the paper **“DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation.”**

## Requirements

**Python=3.7.9**

**PyTorch=1.6.0**

**Scipy = 1.5.2**

**Numpy = 1.19.1**

## Usage

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an 20-core Intel Xeon CPU, 40GB RAM and a Tesla P40 24GB GPU

`CUDA_VISIBLE_DEVICES=0 python3 -u train_rec.py --dataset sport_cloth > sport_cloth.log 2>&1 & `                                                                                                                                                                                                            
