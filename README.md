# Facial-Landmarks

This repository contains 

* Create environment:
```bash
export env_name="facial-landmarks"
conda create -n $env_name python=3.7
conda activate $env_name
conda install --file requirements.txt
```
## Table of content

- [Dataset](#dataset)
- [Training](#train)

<a name="dataset"><h2>Dataset</h2></a>

In the training set of 64000 face images from the VGGFace dataset, the coordinates (x, y)
of 971 key points are known for each face. 
File with coordinates of points for all faces: train / landmarks.csv.

The test sample contains 16000 images.

Download: https://cloud.mail.ru/public/xPaB/twWLKix2c

<a name="train"><h2>Training</h2></a>
```bash
python train.py --name "baseline" --data "PATH_TO_DATA" [--gpu]
example 
python train.py --name "dance" --data169v2 'data/' --gpu -e 20
```

