# import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os, sys
from tqdm import tqdm
from Dev.Analytics.tools.datautils import DataUtils
from fillgaps.proc.gaps import Gaps
from tools.debug import Debug
from fillgaps.neuralnet.deepdenoiser import DeepDenoiser

##### Training set
N      = 1 # number of variations per patient scan
nHoles = 10000 # number of holes per scan
nVal   = 8 # number of patient taken out of training for validation

utils    = DataUtils()
gaps     = Gaps()
debug    = Debug()

# tensors_qmask, headers = utils.load_nii_all("Qmask")
# tensors_qmask          = tensors_qmask.astype(dtype=bool)
# tensor_uniqmask        = gaps.elementwise_or(tensors_qmask)

def create_dataset(tensors,nHoles=100):
    inputs                 = np.zeros((N,)+tensors.shape)
    labels                 = np.zeros((N,)+tensors.shape)
    mean_val = np.nanmean(tensors)
    for id in tqdm(range(N)):
        for idt, _tensor in enumerate(tensors):
            nan_mask = np.isnan(_tensor)
            _tensor[nan_mask] = mean_val
            inputs[id,idt] = gaps.create_cluster(_tensor,K=nHoles)
            labels[id,idt] = _tensor
    inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
    labels = labels.reshape(-1, labels.shape[2], labels.shape[3], labels.shape[4])
    return inputs, labels

def save_data(X_train, Y_train, X_val, Y_val, directory="data", filename="dataset.npz"):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    np.savez(file_path, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
    debug.success("Saved dataset to",filename)


# Example usage
# Assuming X_train, Y_train, X_val, Y_val are defined


## Create dataset
tensors_basic, _      = utils.load_nii_all("Holes",normalization=True)
debug.success("Loaded tensors_basic with",tensors_basic.shape[0],
                 "examples of shape",tensors_basic.shape[1::],
                 "and min-max range:",tensors_basic.min(),tensors_basic.max())
## Padd the with extra 0 to create even numbered dimensions, useful for autoencoder training
tensors_basic         = tensors_basic[:, :-1, 4:-5, :-1].astype(np.float16)
tensors_basic_val     = tensors_basic[0:nVal]
tensors_basic_train   = tensors_basic[nVal::]

X_train, Y_train = create_dataset(tensors_basic_train)
X_val  , Y_val   = create_dataset(tensors_basic_val)

debug.success("Created training set with",X_train.shape[0],
                 "examples of shape",X_train.shape[1::],
                 "min-max range:",X_train.min(),X_train.max())

debug.success("Created validation set with",X_val.shape[0],
                 "examples of shape",X_val.shape[1::])

name = "dataset_nHoles_" + str(nHoles) + ".npz"
# save_data(X_train, Y_train, X_val, Y_val,filename=name)
input_shape = (X_train.shape[1::]+(1,))
# input_shape = (114, 138, 114,1)
ddn = DeepDenoiser()
autoencoder = ddn.build_autoencoder(input_shape)
autoencoder.summary()

history = ddn.train(autoencoder, X_train, Y_train, 
              X_val  , Y_val, 
              epochs=10, batch_size=1)













