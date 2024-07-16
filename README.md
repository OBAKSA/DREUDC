# DREUDC
[ECCV2024] Panel-Specific Degradation Representation for Raw Under-Display Camera Image Restoration

We are currently refining our codes that are to be released. 

## Abstract
Under-display camera (UDC) image restoration aims to restore images distorted by the OLED display panel covering the frontal camera on a smartphone. Previous deep learning-based UDC restoration methods focused on restoring the image within the RGB domain with the collection of real or synthetic RGB datasets. However, UDC images in these datasets exhibit domain differences from real commercial smartphone UDC images while inherently constraining the problem and solution within the RGB domain. To address this issue, we collect well-aligned sensor-level real UDC images using panels from two commercial smartphones equipped with UDC. We also propose a new UDC restoration method to exploit the disparities between degradations caused by different panels, considering that UDC degradations are specific to the type of OLED panel. For this purpose, we train an encoder with an unsupervised learning scheme using triplet loss that aims to extract the inherent degradations caused by different panels from degraded UDC images as implicit representations. The learned panel-specific degradation representations are then provided as priors to our restoration network based on an efficient Transformer network. Extensive experiments show that our proposed method achieves state-of-the-art performance on our real raw image dataset and generalizes well to previous datasets.

## Overview of Our Proposed Method (DREUDC)

<p align="center"><img src="figure/overview_encoder.PNG" width="900"></p>

Our proposed method on training an encoder to learn panel-specific degradation representations. The encoder remains frozen after this stage.

<p align="center"><img src="figure/overview_framework.PNG" width="900"></p>

Overall framework of DREUDC, which utilizes the panel-specific degradation representation encoder by embedding the encoder output to the restoration network.

## Datasets
Our collected dataset can be downloaded via this link. 
* [Google Drive](https://drive.google.com/drive/folders/1k4RQQJhNNKa3J_XC0lI2XGojL8xWCUA9?usp=sharing)
<!-- * [Hugging Face](https://huggingface.co/datasets/obaksa/CommercialUDC) -->

Our dataset can only be used for non-commercial research purposes under the condition of properly attributing the original authors. All the images are collected from DIV2K dataset. The copyright belongs to SK hynix Inc. and the original dataset owners.

## Experimental Results

**Visualized results on our collected dataset, Axon 30 and Z-Fold 3**
<p align="center"><img src="figure/qualitative.PNG" width="900"></p>

## Citation
```
will be updated soon
```
