#rewriting everything for pytorch
import sys
sys.path.append('..') # Add src to path
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from IPython.display import Image

#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from src.display import get_cmap
from src.utils import make_log_dir

# comment these out if you don't have cartopy
import cartopy.feature as cfeature
from src.display.cartopy import make_ccrs,make_animation

from make_dataset import NowcastGenerator,get_nowcast_train_generator,get_nowcast_test_generator

from unet_benchmark import UNet
from unet_benchmark import nowcast_mae, nowcast_mse


#data is already downloaded in the sevir_challenges folder
data_path="../../sevir_challenges"

# Target locations of sample training & testing data
DEST_TRAIN_FILE= os.path.join(data_path,'data/processed/nowcast_training_000.h5')
DEST_TRAIN_META=os.path.join(data_path, 'data/processed/nowcast_training_000_META.csv')
DEST_TEST_FILE= os.path.join(data_path, 'data/processed/nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'data/processed/nowcast_testing_000_META.csv')

# THIS DOWNLOADS APPROXIMATELY 40 GB DATASETS (AFTER DECOMPRESSION)
import boto3
from botocore.handlers import disable_signing
import tarfile
resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir')

print('Dowloading sample training data')
if not os.path.exists(DEST_TRAIN_FILE):
    bucket.download_file('data/processed/nowcast_training_000.h5.tar.gz',DEST_TRAIN_FILE+'.tar.gz')
    bucket.download_file('data/processed/nowcast_training_000_META.csv',DEST_TRAIN_META)
    with tarfile.open(DEST_TRAIN_FILE+'.tar.gz') as tfile:
        tfile.extract('data/processed/nowcast_training_000.h5','..')
else:
    print('Train file %s already exists' % DEST_TRAIN_FILE)
print('Dowloading sample testing data')
if not os.path.exists(DEST_TEST_FILE):
    bucket.download_file('data/processed/nowcast_testing_000.h5.tar.gz',DEST_TEST_FILE+'.tar.gz')
    bucket.download_file('data/processed/nowcast_testing_000_META.csv',DEST_TEST_META)
    with tarfile.open(DEST_TEST_FILE+'.tar.gz') as tfile:
        tfile.extract('data/processed/nowcast_testing_000.h5','..')
else:
    print('Test file %s already exists' % DEST_TEST_FILE)

# Control how many samples are read.   Set to -1 to read all 5000 samples.
N_TRAIN= 100
TRAIN_VAL_FRAC=0.8
N_TEST= 40

# Loading data takes a few minutes
with h5py.File(DEST_TRAIN_FILE,'r') as hf:
    Nr = N_TRAIN if N_TRAIN>=0 else hf['IN_vil'].shape[0]
    X_train = hf['IN_vil'][:Nr]
    Y_train = hf['OUT_vil'][:Nr]
    training_meta = pd.read_csv(DEST_TRAIN_META).iloc[:Nr]
    X_train,X_val=np.split(X_train,[int(TRAIN_VAL_FRAC*Nr)])
    Y_train,Y_val=np.split(Y_train,[int(TRAIN_VAL_FRAC*Nr)])
    training_meta,val_meta=np.split(training_meta,[int(TRAIN_VAL_FRAC*Nr)])

with h5py.File(DEST_TEST_FILE,'r') as hf:
    Nr = N_TEST if N_TEST>=0 else hf['IN_vil'].shape[0]
    X_test = hf['IN_vil'][:Nr]
    Y_test = hf['OUT_vil'][:Nr]
    testing_meta=pd.read_csv(DEST_TEST_META).iloc[:Nr]
    
X_train = X_train.transpose(0,3,1,2)
X_val = X_val.transpose(0,3,1,2)
Y_train = Y_train.transpose(0,3,1,2)
Y_val = Y_val.transpose(0,3,1,2)
X_test = X_test.transpose(0,3,1,2)
Y_test = Y_test.transpose(0,3,1,2)


