[![DOI](https://zenodo.org/badge/549019105.svg)](https://zenodo.org/badge/latestdoi/549019105)

## : Code for "Self-supervised Co-learning of Uncurated Images and Reports Enables Oversight AI in Radiology"
### Medical X-VL: Medical Domain X-attention Vision-Language model
### Paper link: [https://arxiv.org/abs/2208.05140](https://arxiv.org/abs/2208.05140)

<div align="center">
  <img src="./assets/teaser.png">
</div>

### [Paper] | [Official Pytorch code](https://github.com/sangjoon-park/)


> **Medical X-VL: Medical Domain X-attention Vision-Language model**<br>
>
> *Medical X-VL is a vision-language pre-training model developed to be tailored for the intrinsic properties of the medical domain data. For demo, we provide python codes where you can vision-language pretrain, fine-tune and evaluate for each task, visualize the cross-attention between the words and visual semantics.*

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

https://drive.google.com/file/d/1RKowiRjRCIj6WUlzhFsJsgaA33g9K9l2/view?usp=sharing


### VLP model for abdominal radiographs

https://drive.google.com/file/d/1Y9uc_eVgp0irNE0BUka9_0qbY5urdS6_/view?usp=sharing


## Training the model
### Vision-Language Pre-training
First, download ImageNet-pretrained weights for the visual encoder from this [link](https://github.com/bytedance/ibot). We utilized pre-trained ViT-S/16 model as the visual encoder.
```
>  --config ./configs/Pretrain.yaml --output_dir ./output/
```

### Image-Report retrieval
Our model support zero-shot retrieval for image-to-text and text-to-image retrieval without any fine-tuning step.
```
>  --config ./configs/Retrieval.yaml --output_dir ./output/ --checkpoint /PATH/TO/PRETRAIN/ --evaluate
```

### Report Generation
From the VLP weights, the model can be fine-tuned for the report generation task as below.
```
>  --config ./configs/Generation.yaml --output_dir ./output/ --checkpoint /PATH/TO/PRETRAIN/
```

After fine-tuning, inference can be done as below.
```
>  --config ./configs/Generation.yaml --output_dir ./output/ --checkpoint /PATH/TO/FINETUNE/ --evaluate
```

### Vision-Question Answering (VQA)
From the VLP weights, the model can be fine-tuned for the VQA task as below.
```
>  --config ./configs/VQA.yaml --output_dir ./output/ --checkpoint /PATH/TO/PRETRAIN/
```

After fine-tuning, inference can be done as below.
```
>  --config ./configs/VQA.yaml --output_dir ./output/ --checkpoint /PATH/TO/FINETUNE/ --evaluate
```

### Error Detection
Human error (patient mismatch, orientation confusion) can be detected without any fine-tuning step, as the model is already trained to correlate the image and report in the pre-training stage.
```
>  --config ./configs/Detection.yaml --output_dir ./output/ --checkpoint /PATH/TO/PRETRAIN/ --evaluate
```

### Visualization
Succesful visualization will show the cross-attention between the words and the visual semantics (image patches) as below.
```
>  --config ./configs/Pretrain.yaml --output_dir ./output/ --checkpoint /PATH/TO/PRETRAIN/ --evaluate
```

<div align="center">
  <img src="./assets/fig_attention.png">
</div>


#### If you have any questions, please contact us via:
depecher@kaist.ac.kr
