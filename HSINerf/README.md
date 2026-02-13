# Misc Notes on Code

These are some notes (and some legacy notes) on the code, how its organized, and where things were changed and why.

## Additional Usage Options

### PCA/SUB

I have now implmented some new options for the model, namely PCA (and "sub" channel models).
To use PCA: `ns-train hsi_nerf --data /.../transforms.json  hsi-data-parser-config  --image_mode pca --n_components 3`.
This will create a NeRF trained on the first 3 PCA components (sorted by eigenvalues of course).
You can also do `--image_mode sub` to train on the find `n_components` channels sorted by the absorption spectrum of the gas.
This isn't really recommended since it is gas dependent.

# Technical Details

Here I'll talk about what I've learned about this code base, what each file is for, and talk about some of the important functions that can be changed.

### HSI_config.py

This is where you define new models to be registered in nerfstudio so that they can be called using `ns-train`.
This is also where the `hsi_dataparser` is defined, but again, adding it to the pyproject causes problems.

To define a model, pretty much just copy a MethodSpecification, and change whatever you want.
There are lots of options that can be changes, so you'll want to go to the documentation/files for the data manager, the data parser, and model configs to see everything that can be changed.
Note that since our hsi nerf is all based on nerfacto, you'll want to poke through the documentation for nerfacto to see what options are available.
I tend to keep the optimizers and main trainer config stuff the same, and just change the HSIModelConfig options.

Once you've made a new model, for example `hsi_nerf_2 = MethodSpecification(...)`, then you'll add it in the `pyproject.toml`.
Just follow the examples of hsi_nerf for how to add the model.
I believe you then have to build the package again with `pip install -e .`.

### HSI_datamanager.py

This is where extra processing steps related to data can be added.
For example, I updated the `setup_train` function, which is called to create the training set.
I added code to compute PCA on the training, so that it can be used elsewhere if needed.
You'll want to look at the documentation/files of the `VanillaDataManager` to see what other functions and stuff could be updated.

### HSI_dataparser.py

This is a direct copy and modification of the `nerfstudio_dataparser.py` data parser, which is the default used when running nerfacto.
I removed any code that wasn't directly related to options we would use for HSI.
It goes through, creates all the `Camera` options from `transforms.json`, and creates the train/test split.
The only thing that you might want to change here is adding any metadata related to the files/transforms, or changing how the train/test split works.
Currently only even distribution of images is implemented for train/test split.

### HSI_dataset.py

This is where the actual files are read in.
The only thing to change here would be how the images are read (for example changing the units, standardizing, etc.).
Though things like that may also be done in the datamanager I presume.
The only other thing done here is tracking the original file name indicies so that when saving the files (from eval) the file names correspond to the actual images.

### HSI_model.py

This is where we make actual changes to the nerfacto model, and where we'll add many of our modiciations and ideas.

We'll start with the `HSIField`.
This is where the neural network, encodings, and other technical aspects of the model are defined.
Mine is an extension of the `NerfactoField`, where we (currently) just update the MLP head to output 128 channels instead of 3.

The `HSIRenderer` is an extension of the default `RGBRenderer`, where we remove an assertion that there should only be 3 channels, and that the output should be clipped to 0, 1.
Nothing else to change here.

`HSIModel` is where we start piecing things together.
First we override the `self.field` with out `HSIField`, as well as update the "rgb" renderer.
See the `NerfactoModel` to see what else could be changed or set in this phase.

There are two main important functions in the Model that we updated.
First is `get_outputs_for_camera_ray_bundle`.
I believe this is used in several places, but I primarily view this as the main visualization function that the viewer uses for plotting.
We first call the regular `get_outputs_for_camera_ray_bundle`, then we add a few more visualization options.
Specifically a few single channel options, and a simple false color (scalled between 0, 1).
Other visualizations could be added here, for example maybe a PCA visualization, but the "RGB" option in the viewer is actually already a PCA representation of the image.

The second function is `get_image_metrics_and_images`.
If I recall, this is primarily used by `eval` to get the metrics and the rendered image.
I only changed a few things about the original, namely returning separate predicted "rgb" and gt images (note that I typically still use the rgb naming from the code so that minimial code needs to be changed. But just know that "rgb" almost always will be refering to the full channel image).
I also don't calculate lpips.
This function is where we would want to add logic to calculate other metrics, such as MSE, SAM, and whatever else, just add it to the `metrics_dict`.
Also if you want eval to save some other aspect of an image, say a PCA visualization, then add that to the `images_dict`.

### HSI_pipeline.py

I think several things could be added to the pipeline, see `VanillaPipeline` to figure out what could be added or changed.
Primarily this is used to by `ns-eval` to generate the average image metrics and render the testing images.
I added code to run the same "eval" metrics on the training set, just so that we can see how well the model is recreating the training data.
Other metrics and stuff could be added here for output (though they would need to be implemented in the previously mentioned `get_image_metrics_and_images`).


I also updated the `get_eval_image_metrics_and_images` to reorder the axes of the images and standardize to 0, 1.
I believe this function is used by tensorboard to display the changing metrics, and to output some iamges.
However I seem to recall tensorboard doesn't like the outputs anyways, but at least it wont crash.

### HSI_utils.py

This was originally made to make the `get_outputs_for_camera_ray_bundle` more modular, but I haven't really used it.
I suppose other utility functions could be added here if they are used throughout the rest of the code.


# Other Notes

Check out nerfstudio.engine.trainer to see where everything gets setup for training (and the training happens).  

When training is run, it runs the following nested setups:

trainer  
-- pipeline  
---- datamanager  
------ dataparser   
------ train/eval datasets (hsi_dataset)  
---- model  
-- optimizers:  