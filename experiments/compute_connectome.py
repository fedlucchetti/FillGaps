# Importing libraries
import h5py, os, sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import connectomics.similarityM as simM
SCALEID  = "1"
X        = 60
DILATE   = False
FMAP     = "/Users/flucchetti/Documents/Connectome/Data/MRSI_reconstructed/4D_Cr+PCr_test2_basic.nii.gz"
FMAP_DIL = "/Users/flucchetti/Documents/Connectome/Data/MRSI_reconstructed/4D_Cr+PCr_test2_basic_dilated.nii.gz"

if DILATE:
    img = nib.load(FMAP)
    slide = img.get_fdata()[:,:,:,0]
    slide = np.nan_to_num(slide)
    slide = (slide - slide.min()) / (slide.max() - slide.min())
    img_dilated = simM.dilate(img,new_shape = (193, 229, 193),extension_mode="nearest")
    nib.save(img_dilated, FMAP_DIL)
    slide_dilated = img_dilated.get_fdata()[:,:,:,0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # First image
    axs[0].imshow(slide[X,:,:])
    axs[0].set_title('OG')
    # Second image
    axs[1].imshow(slide_dilated[X,:,:])
    axs[1].set_title('Dilated')
    plt.show()

FMAP = FMAP_DIL
# hfName = os.path.join(os.getcwd(), "wm.connatlas." + SCALEID + ".h5")
hfName="/Users/flucchetti/Documents/Connectome/Data/Atlas/probconnatlas/probconnatlas/wm.connatlas.scale1.h5"

simM.bundles_to_connectivity(hfName="probconnatlas/wm.connatlas.scale1.h5",
                             outBasename="outtest/", 
                             scaleId=SCALEID, 
                             mapFilename=FMAP, 
                             voxth=0.1, 
                             subth=0.1, 
                             boolbund=False, 
                             boolforce=False)

# sys.exit()

# Reading the h5 file


connFilename = 'outtest/-connmat-mean-1.h5'
hf = h5py.File(connFilename, 'r')
meanCon       = hf.get('connmat/matrix')

medFilename =  'outtest/-connmat-median-1.h5'
hf = h5py.File(medFilename, 'r')
medianCon       = hf.get('connmat/matrix')

plt.imshow(np.array(meanCon))
plt.colorbar()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# First image
axs[0].imshow(np.array(meanCon))
axs[0].set_title('Mean Conn')
# Second image
axs[1].imshow(np.array(medianCon))
axs[1].set_title('Median Conn')
# Ploting the matrix using matplotlib
# plt.colorbar()
plt.show()