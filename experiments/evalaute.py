# import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os, sys

from tools.utilities import Utilities
from proc.gaps import Gaps

utils = Utilities()
gaps  = Gaps()

tensor, headers = utils.load_nii_all("Qmask")
qmask           = gaps.elementwise_or(tensor.astype(np.int64))

nifti_img = nib.Nifti1Image(qmask.astype(np.float64), np.eye(4))
outpath   = os.path.join(utils.DATAPATH,
                         "MRSI_reconstructed",
                         'qmask_population.nii')
nifti_img.to_filename(outpath)




