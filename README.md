# CVPR2021 - Paper ID 7123
Pytorch implemetnation of **Estimating A Child's Growth Potential From Cephalometric X-Ray Image via Morphology-Aware Interactive Keypoint Estimation**.

## Supplementary video
Our supplementary video is provided in `supplementary_vedio.zip` file.

## Prerequisites
Install dependencies using the `requirements.txt` file.
```
conda install --file requirements.txt
```

## Dataset
Our dataset is not provided due to the confidentiality and privacy of personal data.
Thus, cephlometric X-ray images and the labels are required to run our code. The files should be downloaded in `data` folder.


## Pretrained models
The pretrained model `model.pth` is provided in the `save/pretrained_model` folder.
```
cd save/pretrained_model/
```

## Training
To train and evaluate our model, run the `main.sh` file.
```
cd codes
bash scripts/main.sh
```


