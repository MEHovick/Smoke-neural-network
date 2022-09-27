import os
import sys
import cv2
import torch
import numpy as np
import albumentations as A
import torch.optim as optim
import torch.utils.data as dt
import matplotlib.pyplot as plt
import torchvision.transforms as tf
import segmentation_models_pytorch as smp

from tqdm import tqdm
from imutils import paths
from torchmetrics import JaccardIndex
from sklearn.model_selection import train_test_split

epoch = 1
data_path = "DATASET/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
memory = True if device == 'cuda' else False
print(device)
