[![DOI](https://zenodo.org/badge/549019105.svg)](https://zenodo.org/badge/latestdoi/549019105)

## : Code for "Self-supervised Co-learning of Uncurated Images and Reports Enables Oversight AI in Radiology"
### Medical X-VL: Medical Domain X-attention Vision-Language model
### Paper link: [https://github.com/sangjoon-park/Medical_X-VL](https://github.com/sangjoon-park/Medical_X-VL)

<div align="center">
  <img src="./assets/teaser.PNG">
</div>

### [Paper] | [Official Pytorch code](https://github.com/sangjoon-park/)


> **Medical X-VL: Medical Domain X-attention Vision-Language model**<br>
>
> *Medical X-VL is a vision-language pre-training model developed to be tailored for the intrinsic properties of the medical domain data. For demo, we provide python codes where you can pre-train, fine-tune, evaluate and visualize the cross-attention between the words and visual semantics.*

## System requirements
### General requirements
#### OS
* Ubuntu 20.04

#### Software
* Python 3.8 (tested on)
* Conda
* Pytorch 1.8.0 (tested on)
* CUDA version 11.3 (tested on)

#### Hardware
* CPU or GPU that supports CUDA CuDNN and Pytorch 1.8.
* We tested on GeFore RTX 3090.
* We recommend RAM of more than 32 GB.

## Installation guide
### Instruction
* Install Pytorch and other dependencies. It can be easily installed with requirements.txt file.
```
>  pip install -r requirements.txt
```

## Data preparation
### Downloading data

The open-source datasets used in paper can be obtained from following links.

#### Dataset preparation
* We follow the [MedViLL](https://github.com/SuperSupermoon/MedViLL) to preprocess and split the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [VQA-RAD](https://osf.io/89kps/) datasets. See this [link](https://github.com/SuperSupermoon/MedViLL) for details.
* COVID-19 and normal data can be downloaded in [Brixia](https://brixia.github.io/) and [NIH](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) databases.

Other parts of the institutional data used in this study are not publicly available due to the patient privacy obligation. Interested users can request the access to these data for research, by contacting the corresponding author J.C.Y. (jong.ye@kaist.ac.kr).


## Download pretrained weights
You can download the pretrained weights on the CheXpert dataset in link below, which should be located as,

### VLP model for Chest radiographs

https://drive.google.com/file/d/16y3eJRYQCg-B8rg9eB3XRA-6PcfHCNmA/view?usp=sharing


### VLP model for abdominal radiographs

https://drive.google.com/file/d/16y3eJRYQCg-B8rg9eB3XRA-6PcfHCNmA/view?usp=sharing


## Training a model
First, download ImageNet-pretrained weights for the visual encoder from this [link](https://github.com/bytedance/ibot). We utilized pre-trained ViT-S/16 model as the visual encoder.



```
## Evaluating a model
You can evaluate the model performance (AUC) with the following code.
```
> python eval_finetune.py --name EXP_NAME --pretrained_dir /PATH/FOLD3/checkpoint.pth --data_path /PATH/DATA/ --checkpoint_key student
```

## Visualizing attention
The attentions of Vision transformer model can be visualized with following code.
```
> python visualize_attention.py --pretrained_weights /PATH/FOLD3/checkpint.pth --image_dir /PATH/DATA/ --checkpoint_key student
```
Successful visualization will provide attention maps as below.

<div align="center">
  <img src="./assets/attention.png">
</div>


#### If you have any questions, please contact us via:
depecher@kaist.ac.kr
