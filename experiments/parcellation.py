import numpy as np
from nilearn import plotting, image, datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from scipy.spatial.distance import pdist, squareform

from Dev.Analytics.tools.datautils import DataUtils
utils = DataUtils()


brain_scans = list()
for i in range(40):
    scan= utils.load_nii("Holes",i,metabolic_str=None,normalization=True,rawnii=True)
    data = scan.get_fdata()
    data[data == -1] = 0
    scan = nib.Nifti1Image(data, affine=scan.affine)
    brain_scans.append(scan)

#  Load an Atlas for Parcellation
atlas_data     = datasets.fetch_atlas_aal()
atlas_filename = atlas_data.maps
labels         = atlas_data.labels

#  Parcellation and Signal Extraction
masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                           standardize=True, 
                           memory='nilearn_cache')

# Extract the mean signal for each ROI in each scan
extracted_features = [masker.fit_transform(scan) for scan in brain_scans]


# Compute the similarity matrix (example using Pearson correlation)
n_regions           = extracted_features[0].shape[1]
n_scans             = len(extracted_features)
similarity_matrices = np.zeros((n_scans, n_regions, n_regions))



for i, features in enumerate(extracted_features):
    features = features[0,:]
    for j in range(n_regions):
        for k in range(n_regions):
            if j != k:
                r = (features[j] - features[k])**2
                similarity_matrices[i, j, k] = r

similarity_matrices = 1 - similarity_matrices/similarity_matrices.max()



######################### Plot results #########################

# Assuming you have the atlas nifti image and labels
# Assuming similarity_matrices is already computed and has shape (n_scans, n_regions, n_regions)
mean_similarity_matrix = np.mean(similarity_matrices, axis=0)
region_mean_similarity = np.mean(mean_similarity_matrix, axis=1)

# Load the atlas and get its data
#atlas_data = fetch_atlas_basc_multiscale_2015()
#atlas_filename = atlas_data.maps
#labels = atlas_data.labels
atlas_img = image.load_img(atlas_filename)
atlas_data_array = atlas_img.get_fdata()

# Create an empty image with the same shape as the atlas
similarity_img_data = np.zeros_like(atlas_data_array)

# Assign the mean similarity value to each ROI
for i, label in enumerate(labels):
    similarity_img_data[atlas_data_array == label] = region_mean_similarity[i]


similarity_img = nib.Nifti1Image(similarity_img_data, affine=atlas_img.affine)

# Plot the mean similarity overlay
plotting.plot_stat_map(similarity_img, bg_img=plotting.load_mni152_template(), 
                       title="Mean ROI Similarity", display_mode='ortho', 
                       cut_coords=[-20, -10, 10, 20], cmap='hot')