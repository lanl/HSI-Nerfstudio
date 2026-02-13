


import torch
import spectral.io.envi as envi
import numpy as np
import re

from rich.console import Console
CONSOLE = Console(width=120)

from nerfstudio.data.datasets.base_dataset import InputDataset

from HSINerf.HSI_utils import ACE_det




class HSIDataset(InputDataset):
    def __init__(self, dataparser_outputs, scale_factor = 1, metadata={}):
        super().__init__(dataparser_outputs, scale_factor)
        # Get original index's
        self.og_idx = [int(re.search(r"MakoSpectrometer-t(\d+)\.img\.hdr", str(fname)).group(1)) for fname in self.image_filenames]

        self.image_mode = metadata.get("image_mode", "full")
        self.n_components = metadata.get("n_components", 3)
        self.n_channels = metadata.get("n_channels", 128)
        self.pca = metadata.get("pca", None)
        self.standardizer = metadata.get("standardizer", None)
        self.targ_sig = metadata.get("targ_sig", None)
        self.det_thresh = metadata.get("det_thresh", .6)


        # Add this info here so that the HSI Model can access it
        self.metadata['pca'] = self.pca
        self.metadata['standardizer'] = self.standardizer
        self.metadata['targ_sig'] = self.targ_sig
        self.metadata['det_thresh'] = self.det_thresh
        self.metadata['patch_size'] = metadata.get("patch_size")

    def __getitem__(self, image_idx):
        # read in envi and make it a torch float32
        filename = self.image_filenames[image_idx]
        img = envi.open(filename).asarray()
        img = np.array(1e6*img, dtype="float32")  # Converts the units to micro flicks
        
        og_image = torch.from_numpy(img).clone()  # default to not cause collate problems
        # og_image = torch.from_numpy(np.array([0]))
        if self.image_mode == 'pca':
            og_image = torch.from_numpy(img)
            og_shape = img.shape
            img = img.reshape((-1, og_shape[-1]))
            img = self.pca.transform(img)
            img = img.reshape((og_shape[0], og_shape[1], self.n_components))
        
        if self.image_mode == 'full' and self.standardizer is not None:
            img = self.standardizer.transform(img.reshape(-1, self.n_channels))
            img = img.reshape(og_image.shape)

        # Calculate GT detection (or load in corresponding detection map)
        ace_img = ACE_det(og_image, self.targ_sig)
        det_mask = ace_img > self.det_thresh

        data = {
            "image_idx": image_idx,
            "image": torch.from_numpy(img),
            "og_image": og_image,
            'ACE_img': ace_img,
            'det_mask': det_mask
        }
        return data
