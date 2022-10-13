from PIL import Image
import numpy as np
import torchvision
import torch

COLOR_JITTER = torchvision.transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)

def image_numpy_to_tensor(np_image, is_eval):
    """Function to convert image numpy array to PyTorch tensor

    Parameters
    ----------
    np_image : numpy array
        An numpy array contains a pillow image

    is_eval : bool
        If true, it returns a 4D tensor
        If false, it returns a 3D tensor

    Returns
    -------
    tensor_image:
        A tensor contains PyTorch image
    """
    np_image = np_image.copy().transpose(2, 0, 1)
    tensor_image = torch.from_numpy(np_image).type(torch.float).cuda()
    tensor_image = tensor_image.reshape(
        1, *tensor_image.shape) if is_eval else tensor_image
    return tensor_image

def raw(image, is_eval=False, apply_color_jitter=True):
    """Function to preprocess images without other preprocessing techniques, such as stain normalization

    Parameters
    ----------
    image : Pillow image
        A pillow image contains a pillow image

    is_eval : bool
        If true, it returns a 4D tensor
        If false, it returns a 3D tensor

    Returns
    -------
    tensor_image:
        A tensor contains PyTorch image
    """
    np_image = np.asarray(COLOR_JITTER(image)).copy(
    ) if apply_color_jitter else np.asarray(image).copy()
    np_image = (np_image - 128.) / 128.
    image_tensor = image_numpy_to_tensor(np_image, is_eval)
    return image_tensor

def ndarray_image_to_tensor(image):
    image = (image - 128.) / 128.
    image = image.transpose(2, 0, 1)
    tensor_image = torch.from_numpy(image).type(torch.float)
    return tensor_image
