# import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os, sys
from tqdm import tqdm
from fillgaps.tools.utilities import Utilities
from fillgaps.proc.gaps import Gaps
from fillgaps.tools.messages import MessagePrinter
from fillgaps.neuralnet.deepdenoiser import DeepDenoiser

##### Training set
N      = 5 # number of variations per patient scan
nHoles = 100 # number of holes per scan
nVal   = 8 # number of patient taken out of training for validation

utils    = Utilities()
gaps     = Gaps()
msgprint = MessagePrinter()

# tensors_qmask, headers = utils.load_nii_all("Qmask")
# tensors_qmask          = tensors_qmask.astype(dtype=bool)
# tensor_uniqmask        = gaps.elementwise_or(tensors_qmask)

def create_dataset(tensors):
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
    return inputs.astype(np.float16), labels.astype(np.float16)

def save_data(X_train, Y_train, X_val, Y_val, directory="data", filename="dataset.npz"):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    np.savez(file_path, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)
    msgprint.success("Saved dataset to",filename)


# Example usage
# Assuming X_train, Y_train, X_val, Y_val are defined


## Create dataset
tensors_basic, _      = utils.load_nii_all("Basic")
## Padd the with extra 0 to create even numbered dimensions, useful for autoencoder training
tensors_basic         = tensors_basic[:, :-1, 4:-5, :-1].astype(np.float16)
tensors_basic_val     = tensors_basic[0:nVal]
tensors_basic_train   = tensors_basic[nVal::]

X_train, Y_train = create_dataset(tensors_basic_train)
X_val  , Y_val   = create_dataset(tensors_basic_val)

msgprint.success("Created training set with",X_train.shape[0],
                 "examples of shape",X_train.shape[1::])

msgprint.success("Created validation set with",X_val.shape[0],
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

# q = input("Do you want to safe the processed population qmask? [y,n]")
# if q=="y":
#     nifti_img = nib.Nifti1Image(tensor_uniqmask.astype(np.float64), np.eye(4))
#     outpath   = os.path.join(utils.DATAPATH,
#                             "MRSI_reconstructed",
#                             'qmask_population.nii')   
#     nifti_img.to_filename(outpath)
#     msgprint.success("Saved to " + outpath)
# nholes_dist = np.zeros(np.shape(tensors_qmask)[0])
# tensors_gaps = np.zeros(tensors_qmask.shape).astype(dtype=bool)
# for idt, tensor in enumerate(tensors_qmask):
#     com = np.bitwise_and(tensor,tensor_uniqmask)
#     tensors_gaps[idt] = np.bitwise_not(com)
#     nholes_dist[idt] = np.prod(tensor_uniqmask.shape) - tensors_gaps[idt].sum()











