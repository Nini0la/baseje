Table of Contents
<!-- TOC -->

- [Setup](#setup)
- [Dataset](#dataset)
- [Notes](#notes)

<!-- /TOC -->
# Setup
- Set up a virtual environment
```
$ python -m venv env
```
- Install requirements
```
$ pip install -r requirements.txt 
```
# Dataset
- This project uses the same dataset as the previous Mask R-CNN project in tensorflow. Your directory structure should be similar to the one shown below.
```
├───data
│   ├───test
│   ├───train
│   └───val
├───env
├───src
├───.gitignore
├────cardamage.ipynb
├───README.md
└───requirements.txt
```
- Run create_dataset in terminal
```
$ cd src
$ python create_dataset.py
```
# Notes
- To install PyTorch on Windows, no CUDA:
  ```
  $ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  ```
