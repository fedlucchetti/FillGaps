import torch
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    ToTensor,
    Resize,
    ScaleIntensity,
)
from monai.inferers import sliding_window_inference
from monai.transforms import Resize

from fillgaps.tools.utilities import Utilities
utils = Utilities()


# Load your data
tensor3D, _ = utils.load_nii("Basic", 1)
# If tensor3D is a numpy array
if isinstance(tensor3D, np.ndarray):
    tensor3D = torch.from_numpy(tensor3D)

# Add a batch dimension
tensor3D = tensor3D.unsqueeze(0)  # Shape becomes [1, D, H, W]

# # Define transforms
# transform = Compose([
#     ScaleIntensity(),
#     Resize(spatial_size=(128, 128, 64)),
#     ToTensor()
# ])

# Apply the transforms to your tensor
resize_transform = Resize(spatial_size=(128, 128, 64))

# Apply the transform
tensor3D_resized = resize_transform(tensor3D)
# Load a pre-trained UNETR model
model = UNETR(
    in_channels=1,
    out_channels=2,  # Assuming binary classification
    img_size=(128, 128, 64),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed='perceptron',
    norm_name='instance',
    conv_block=True,
    res_block=True,
).to(torch.device('cpu'))

# Load the pre-trained weights (if you have them)
# model.load_state_dict(torch.load('path_to_pretrained_model.pth'))
tensor3D_resized[np.isnan(tensor3D_resized)] = 0
tensor3D_resized/
# Set model to evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    outputs = sliding_window_inference(tensor3D_resized.unsqueeze(0), (128, 128, 64), 4, model)
    # Assuming binary classification, get the class with the higher probability
    result = torch.argmax(outputs, dim=1).squeeze(0)



result_np = result.detach().cpu().numpy()

# Visualize in 3D
viewer = itkwidgets.view(image=result_np)
viewer