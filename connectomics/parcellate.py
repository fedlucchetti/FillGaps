import nibabel as nib
import numpy as np

from tqdm import tqdm

from scipy.ndimage import zoom




class Parcellate:
    def __init__(self) -> None:
        pass

    def dilate(img,new_shape = (193, 229, 193), extension_mode="constant"):
        # Calculate the zoom factors for each dimension
        data = np.nan_to_num(img.get_fdata())
        data = (data - data.min()) / (data.max() - data.min())
        img  = nib.nifti1.Nifti1Image(data, img.affine,header=img.header)
        data = img.get_fdata()
        # Calculate the zoom factors for each dimension
        zoom_factors = (new_shape[0] / data.shape[0], new_shape[1] / data.shape[1], new_shape[2] / data.shape[2])

        # Perform the zoom operation (interpolation)
        new_shape = new_shape+(data.shape[-1],)
        resized_data = np.zeros(new_shape)
        for i in tqdm(range(data.shape[-1])):
            resized_data[:,:,:,i] = zoom(data[:,:,:,i], zoom_factors, order=3,mode=extension_mode)  # 'order=3' for cubic interpolation

        resized_data[resized_data<0.001] = 0.0
        # Create a new NiBabel image from the resized data
        new_img = nib.nifti1.Nifti1Image(resized_data, img.affine,header=img.header)

        return new_img