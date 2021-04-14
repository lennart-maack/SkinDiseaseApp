import os
import glob
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from wtfml.utils import EarlyStopping



# Part 1 - setting the paths to training, validation images and according the ground truth files
"""
You need to split the image data (which you downloaded from the ISIIC website) according to the gt_train and gt_val csv files into two folders
The paths to these folders need to be put into the variables input_path_training_img and input_path_val_img. Spliting the image data can be done 
with val_split.py

Additionally, there is also a resize.py file in the gitlab repository, which resizes your images to the wanted size.

If you have question regarding this CNN script, the val_split.py file or the resize.py file, just text me.
""" 

input_path_training_img = ""
input_path_val_img = ""
input_path_predict_img = ""

path_train_gt = ""
path_val_gt = ""

path_predict_gt = ""


# Part2 Setting Hyperparameters

H = 512
W = 512

num_epochs = 9

batch_size = 8

patience_for_early_stopping = 5

learning_rate = 0.0001
