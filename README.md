# BIPS: Bi-modal Indoor Panorama Synthesis via Residual Depth-aided Adversarial Learning (ECCV 2022)

## Abstract
Providing omnidirectional depth along with RGB information is important for numerous applications, \eg, VR/AR.
However, as omnidirectional RGB-D data is not always available, synthesizing RGB-D panorama data from limited information of a scene can be useful.
Therefore, some prior works tried to synthesize RGB panorama images from perspective RGB images; however, they suffer from limited image quality and can not be directly extended for RGB-D panorama synthesis.
In this paper, we study a new problem: RGB-D panorama synthesis under the arbitrary configurations of cameras and depth sensors.
Accordingly, we propose a novel bi-modal (RGB-D) panorama synthesis (BIPS) framework. 
Especially, we focus on indoor environments where the RGB-D panorama can provide a complete 3D model for many applications.
We design a generator that fuses the bi-modal information and train it with residual-aided adversarial learning (RDAL).
RDAL allows to synthesize realistic indoor layout structures and interiors by jointly inferring RGB panorama, layout depth, and residual depth. 
In addition, as there is no tailored evaluation metric for RGB-D panorama synthesis, we propose a novel metric to effectively evaluate its perceptual quality.
Extensive experiments show that our method synthesizes high-quality indoor RGB-D panoramas and provides realistic 3D indoor models than prior methods.


## Requirements
Create the environment from the "environment.yml" file:

```bash
conda env create -f environment.yml
```

Then activate the conda env "BIPS":

```bash
conda activate BIPS
```

## Data Preparation
We use Structured 3D RGB panorama, layout depth panorama (D_ini), final depth panorama (D) for training and testing.
Residual depth is automatically calculated by subtracting layout depth panorama from final depth panorama in the dataloader.

```bash
dataset
|_ train
   |_ RGB
      |_ RGB00020.png
      |_ ...
   |_ D
      |_ Depth00020.png
      |_ ...
   |_ D_ini
      |_ D_ini00020.png
      |_ ...
|_ val
   |_ RGB
   |_ D
   |_ D_ini
|_ test
   |_ RGB
   |_ D
   |_ D_ini
```

## BIPS Framework 
### Training Generator and Discriminator
Train the overall BIPS framework by entering:

```bash
python residual_trainer.py --name [model_name]
```
Synthesized RGB-D panorama results and input masks are saved every 100 iterations of training in the "repo/train" and "repo/val" folder.\
The weight of the generator and discriminator are saved every 1 epoch of training in the "repo/model" folder.\
Since the maximum file size of supplementary material is 100Mb, we couldn't attach the pretrained weight of the generator and discriminator.\
(Weight of generator is around 580Mb and weight of discriminator is around 100Mb).\
We trained the BIPS framework with the entire dataset for 100 epochs to get the results in the main paper.

## FAED Calculation
### 1. Training Auto-Encoder-Decoder Network
```bash
cd FAED
python train.py --name [autoencoder_model_name]
```
Since the maximum file size of supplementary material is 100Mb, we couldn't attach the pretrained weight of the auto-encoder.\
(Weight of auto-encoder is around 100Mb).\
We trained an auto-encoder-decoder network with the entire dataset for 60 epochs.
Since the auto-encoder-decoder network doesn't require layout depth panorama, you can download the publicly available Structured3D dataset and use it for training.

### 2. Exporting Distribution of Data
```bash
python test.py --name [statistic_name]
```
You need to export two distributions of data for comparison.

### 3. Calculating FAED Score
```bash
python FID.py
```
FID.py takes two data distributions inside the code and outputs the FAED score.
### 
## TBD
pretrained weights. 