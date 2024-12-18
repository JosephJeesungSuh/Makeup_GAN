import glob
import os
import pathlib
import random
from typing import List

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class Makeup_dataset(Dataset):

    def __init__(self, data_path, transform, transform_mask):
        self.data_path: pathlib.Path = data_path
        self.transform = transform
        self.transform_mask = transform_mask
        self.src_files : List[str] = glob.glob( # all non-makeup images
            os.path.join(data_path, 'images', 'non-makeup', '*.png')
        )
        self.ref_files : List[str] = glob.glob( # all makeup images
            os.path.join(data_path, 'images', 'makeup', '*.png')
        )
        self.src_mask_files = [ # mask of non-makeup images, same ordering
            os.path.join(data_path, 'segs', 'non-makeup', image_path.split('/')[-1])
            for image_path in self.src_files
        ]
        self.ref_mask_files = [ # mask of makeup images, same ordering
            os.path.join(data_path, 'segs', 'makeup', image_path.split('/')[-1])
            for image_path in self.ref_files
        ]

    def __getitem__(self, index):
        """ Get a pair of random images drawn from source and reference bank. """
        src_idx = random.choice(range(len(self.src_files))) # just random index worked.
        ref_idx = random.choice(range(len(self.ref_files))) # just random index worked.
        src_mask = Image.open(self.src_mask_files[src_idx]).convert("RGB")
        ref_mask = Image.open(self.ref_mask_files[ref_idx]).convert("RGB")
        src_img = Image.open(self.src_files[src_idx]).convert("RGB")
        ref_img = Image.open(self.ref_files[ref_idx]).convert("RGB")
        return (
            self.transform(src_img), 
            self.transform(ref_img), 
            self.transform_mask(src_mask), 
            self.transform_mask(ref_mask),
        )

    def __len__(self):
        """ Length of the makeup dataset. """
        return max(len(self.src_files), len(self.ref_files))


def custom_toTensor(pil_image):
    """
    Convert a PIL Image to a PyTorch tensor.
    """
    modes = {'I': np.int32, 'I;16': np.int16}
    nchannel = {'YCbCr': 3, 'I;16': 1}.get(pil_image.mode, len(pil_image.mode))

    dtype = modes.get(pil_image.mode, torch.uint8)
    img = (
        torch.from_numpy(np.array(pil_image, dtype, copy=False))
        if pil_image.mode in modes
        else torch.ByteTensor(torch.ByteStorage.from_buffer(pil_image.tobytes()))
    )
    img = img.view(
        pil_image.size[1], pil_image.size[0], nchannel
    ).permute(2, 0, 1).contiguous()
    
    return img.float()


def load_data(data_dir, batch_size):
    """
    Get image and return a dataloader.
    """
    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)
    print("load_data(): Data directory:", str(data_dir))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'{data_dir} does not exist')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=Image.NEAREST, # preserve label value
        ),
        custom_toTensor,
    ])
    dataset_train = Makeup_dataset(data_dir, transform, transform_mask)
    dataloader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    print("load_data(): Data loader ready.")
    return dataloader