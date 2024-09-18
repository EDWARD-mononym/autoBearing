import math
import numpy as np
from pathlib import Path
import requests
import torch
from tqdm.auto import tqdm

def set_seed_and_deterministic(seed):
    """
    Sets the seed for NumPy and PyTorch and makes PyTorch operations deterministic.

    Parameters:
    seed (int): The seed value to be set for reproducibility.
    """
    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in PyTorch
    # Note: This might impact performance and is not guaranteed for all operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_file(url, download_path, file_name=None):
        N_tries = 0
        downloaded = False
        if not file_name:
            file_name = Path(url).name
        file_download_path = download_path.joinpath(file_name)

        while N_tries < 5:
            try:
                response = requests.get(url)
                file_download_path.write_bytes(response.content)
                downloaded = True
                break

            except requests.exceptions.ChunkedEncodingError:
                N_tries += 1

        if not downloaded:
            raise Exception(f'Failed to download {file_name} after 5 tries, please rerun the code at another time')

def sliding_window_subsample(tensor_x, tensor_y, window_size, step):
        if len(tensor_x.shape) == 2:
            tensor_x.unsqueeze(1)
        tensor_x = tensor_x.unfold(2, window_size, step)
        B, C, W, L = tensor_x.size() # Get the tensor dimensions for reshaping
        tensor_x = tensor_x.reshape(B*W, C, L)
        tensor_y = tensor_y.unsqueeze(1).repeat(1, W).reshape(B*W)
        return tensor_x, tensor_y

def normalise_tensor(tensor):
    std, mean = torch.std_mean(tensor, dim=2, keepdim=True)
    return (tensor - mean) / std

def subsample_fewshots(tensor_x, tensor_y, few_shot_size):
    skip_every_n = math.ceil(1.0/few_shot_size)
    return tensor_x[::skip_every_n], tensor_y[::skip_every_n]