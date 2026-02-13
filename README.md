# HSI Nerfstudio

This folder contains the code to implement the HSI NeRF capabilities into Nerfstudio.
Currently this implements the following HSI NeRF ideas:

1. Mip-NeRF and Nerfacto for HSI
2. Multi-channel Density
3. Geometry Regularization

## Conda Environment

The HSI models need to be "registered" in nerfstudio before they can be used.
For detail on adding a new method specification (version of the model) click [here](https://docs.nerf.studio/developer_guides/new_methods.html).
Just follow the [nerfstudio](https://docs.nerf.studio/quickstart/installation.html) installation page to make the right conda environment.

```bash 
conda create --name hsi-nerf -y python=3.8
conda activate hsi-nerf
python -m pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
pip install spectral
pip install tensorboard
conda install -c conda-forge ffmpeg
pip install -e .
```

To register the methods, run `pip install -e .`.

## Data (transforms.json)

The data needed to run this code is found on [Zenodo](https://doi.org/10.5281/zenodo.18626884). 
Download the data and unzip it so that the file structure looks something like (i.e., just unzip in your repo):  

<pre>
hsi-nerf
├── data
│   └── complex_facility
│       ├── camera_paths
│       └── Images
├── HSINerf
│   └── scripts
├── notebooks
├── outputs
│   └── complex_facility
│       ├── ...
├── renders
│   └── ...
└── scripts
README.md
...
</pre>

**NOTE: if the data is downloaded from the previous link, the transforms are already made, if not, read the following:**

Before being able to run the model, the transforms.json file needs to be created.
The transforms.json is already made, but if a new one needs to be created, run `scripts/format_transform.sh`.
Note that we have the filenames as just the file name, not the full file path.
At the very end there is a datadir which points to the directory the data is stored.



## Usage (Training)

The following is a basic example usage of our model, demonstrating how to train, evaluation, view, and render the results from NeRF.

```bash
# Train Mip-NeRF model
CUDA_VISIBLE_DEVICES=0 hsi-train hsi_mipnerf_MD_GR --data data/complex_facility/transforms.json --timestamp readme_test hsi-data --train_split_fraction 20 --train_seed 0

# Evaluate Mip-NeRF Model
CUDA_VISIBE_DEVICES=0 hsi-eval --base-dir outputs/complex_facility/hsi_mipnerf_MD_GR/readme_test

# open viewer to generate render command (and view the model results)
CUDA_VISIBLE_DEVICES=0 hsi-viewer --load-config outputs/complex_facility/hsi_mipnerf_MD_GR/readme_test/config.yml

# Example of rendering a path
CUDA_VISIBLE_DEVICES=0 hsi-render camera-path --load-config outputs/complex_facility/hsi_mipnerf_MD_GR/readme_test/config.yml --rendered-output-names Falsecolor ACE depth --output-path renders/complex_facility/hsi_mipnerf_MD_GR/readme_test.mp4 --camera-path-filename data/complex_facility/camera_paths/Drone_Path_48fov.json
```

The following are other common options for running.

- For basic running: `hsi-train hsi_mipnerf --data /path_to_transforms/transforms.json`
- To continue training an existing model: `hsi-train hsi_mipnerf --load-config /path_to_output/config.yml`
- In case tensorboard isn't running (though it should by default): `hsi-train hsi_mipnerf --data /path_to_transforms/transforms.json --vis viewer+tensorboard`. Tensorboard can then be run (using VScode) with `tensorboard --logdir /path_to_output/Date(like 2025-02-19_114609) --port=6007`.
- Changing train/test size: `hsi-train hsi_mipnerf --data /path_to_transforms/transforms.json    hsi-data --fixed_eval_fraction .2`. Currently this will use the same (randomly selected) evaluation set.
- Changing model options `hsi-train hsi_mipnerf --data ... --pipeline.model.use_multi_channel_density False` for example. You can always use `hsi-train hsi_mipnerf --help`.

To evaulate the models, use:

- `hsi_eval --base-dir /path_to_output/`

Note that during training the newest model (saved as `step-xxxxxxxxx.ckpt`) and the best evaluation model (`best_eval_psnr.ckpt`) are tracked.  
When running hsi_eval, it will create an `hsi-eval/best` and `hsi-eval/latest` folder where the evaluation results will be saved.  
In the folder there will be rendered images, as well as other images like the ace detection score image.  
There are other options like `--get_train_metrics` which will also get metrics on the training images, `--save_keys img ace_img det_mask` (these output options are found in the HSI_model_mip.py, the get_image_metrics_and_images function).  

After training, you can view the model and generate rendering options using `hsi-viewer --load-config /path-to-output/config.yml`.  
You can then use this to generate render paths.  
From the rendered path, swap ns-render for hsi-render (it will use the best_eval_psnr model by default).

The current set of rendering .json files uses the default 70 FOV, but to match the training images, you'll need to use an FOV of 48.4 (the `Drone_path_48fov.json` is the only pre-made path that uses 48 FOV).

<video 
  src="renders/complex_facility/hsi_mipnerf_MD_GR/readme_test.mp4" 
  width="640" 
  height="360" 
  loop
  controls>
</video>

### Paper Replication

The code here can also reproduce the results from the preprint [here](https://doi.org/10.36227/techrxiv.176127454.46552904/v1).
To recreate the models and renders related to the paper, first run `scripts/train_all_models.sh` to train all the needed models (including those for the ablation study).
Read documentation in the script for what options to change as far as runtime options (shouldn't change seeds, methods, etc. to match paper).
Then, `render_example_models.sh` can generate full NeRF renders for the specific model examples demonstrated in the paper (and found in renders).
Then, the `Paper_plots.ipynb` can be run to reproduce the static plots found in the paper, as well as the tables.
*NOTE: there are pytorch functions that are NOT deterministic! Therefore it will be impossible to reproduce the exact numbers found in the paper. What is provided the code that was used to produce the exact numbers in the paper.*




## Acknowledgement

See [RegNeRF](https://m-niemeyer.github.io/regnerf/) for the original reference for our geometry regularization.  
See [Multi-Channel-Hyperspectral-NeRF](https://github.com/hhdgq/Multi-Channel-Hyperspectral-NeRF-) for the original reference for our multi-channel density NeRF.  
Note that our coded versions of geometry regularization and multi-channel densities are not copies of the original code.  
ChatGPT was used in the development process of this package, including helping to write and debug functions.

LANL O#: O5204

*© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.*


