import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fillgaps.tools.utilities import Utilities
from fillgaps.proc.gaps import Gaps
from fillgaps.tools.debug import Debug
import time, os
from scipy.stats import binomtest

##### Training set
N        = 5 # number of variations per patient scan
arr_nHoles = np.array([10,50,100,500,1000,2000,5000,10000])
arr_margin_percent = np.array([5])
utils    = Utilities()
gaps     = Gaps()
debug = Debug()

## Create dataset
tensors_basic, _      = utils.load_nii_all("Basic")
# tensors_qmask, _      = utils.load_nii_all("Qmask")


tensors_basic         = tensors_basic[:, :-1, 4:-5, :-1]
tensors_basic         = tensors_basic/np.nanmax(tensors_basic)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for idm,crop_area_pc in enumerate(tqdm(arr_margin_percent)):
    debug.separator()
    debug.info("Crop area percentage",crop_area_pc,"%")
    loss         = np.zeros([len(arr_nHoles),tensors_basic.shape[0]])
    var          = np.zeros([len(arr_nHoles),tensors_basic.shape[0]])
    elapsed_time = np.zeros([len(arr_nHoles),tensors_basic.shape[0]])
    for idh,nHoles in enumerate(tqdm(arr_nHoles)):
        X_train, Y_train      = gaps.create_dataset(tensors_basic,nHoles=nHoles)
        for idt, tensor in enumerate(tqdm(X_train)):
            start_time = time.time()  # Record start time
            interpolated_tensor = gaps.interpolate_missing_values(tensor.copy(),margin_percent=crop_area_pc)
            elapsed_time[idh,idt] = time.time() - start_time
            l = gaps.reconstruction_error(tensor.copy(), interpolated_tensor, Y_train[idt])
            loss[idh,idt] = l
            # loss[idh,idt] = np.mean(interpolated_tensor-tensor)
            var[idh,idt]  = np.std(interpolated_tensor-tensor)
        debug.success("Reconstructed with loss for nHoles",nHoles,"with L=",loss[idh].mean())


    # ax1.fill_between(arr_nHoles, loss.mean(axis=1)+var.mean(axis=1),
    #                 loss.mean(axis=1)-var.mean(axis=1),
    #                 alpha=0.23)
    ax1.plot(arr_nHoles, loss.mean(axis=1),label=f"Crop % {crop_area_pc}, $\pm 1 \sigma$")
    ax1.set_xlabel("Size of Holes",fontsize=16)
    ax1.set_ylabel("Reconstruction Error [%]",fontsize=16)

    ax2.plot(arr_nHoles,  np.around(elapsed_time.mean(axis=1), decimals=2),label = f"Crop % {crop_area_pc}")
    ax2.set_xlabel("Size of Holes")
    ax2.set_ylabel("Computation Time [s] ")

# Adjust layout
ax1.grid()
ax2.grid()

ax1.legend()
ax2.legend()

plt.tight_layout()
path = os.path.join(utils.DEVPATH,"FillGaps","results","interpolate3D_nearest_neighbor.pdf")
plt.savefig(path, format='pdf', bbox_inches='tight')

plt.show()











# q = input("Do you want to safe the processed population qmask? [y,n]")
# if q=="y":
#     nifti_img = nib.Nifti1Image(tensor_uniqmask.astype(np.float64), np.eye(4))
#     outpath   = os.path.join(utils.DATAPATH,
#                             "MRSI_reconstructed",
#                             'qmask_population.nii')   
#     nifti_img.to_filename(outpath)
#     debug.success("Saved to " + outpath)
# nholes_dist = np.zeros(np.shape(tensors_qmask)[0])
# tensors_gaps = np.zeros(tensors_qmask.shape).astype(dtype=bool)
# for idt, tensor in enumerate(tensors_qmask):
#     com = np.bitwise_and(tensor,tensor_uniqmask)
#     tensors_gaps[idt] = np.bitwise_not(com)
#     nholes_dist[idt] = np.prod(tensor_uniqmask.shape) - tensors_gaps[idt].sum()











