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

# def download(url, download_path, chunk_size=1024):
#     resp = requests.get(url, stream=True)
#     total = int(resp.headers.get('content-length', 0))
#     with open(download_path, 'wb') as file, tqdm(
#         desc=download_path,
#         total=total,
#         unit='iB',
#         unit_scale=True,
#         unit_divisor=1024,
#     ) as bar:
#         for data in resp.iter_content(chunk_size=chunk_size):
#             size = file.write(data)
#             bar.update(size)