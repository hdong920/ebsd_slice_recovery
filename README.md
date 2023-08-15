# EBSD Slice Recovery

This repository contains code to train an 3D axial transformer to recover missing EBSD slices, as described in \[ADD PAPER\]. The `src` folder contains code files for the model, projection algorithm, and training procedure. Due to the size of the trained model and data files, they can be found in this [Google Drive](https://drive.google.com/drive/folders/1S-jKZ7wxIT4ra4q5VJ_vEl4rO3yYQSgQ?usp=sharing). The naming convention for the data files is `train_mu<mean grain size>_sig0-4_trans<mean transformations per grain>.dream3d` for training volumes and `val_mu<mean grain size>_sig0-4_trans<mean transformations per grain>_<volume number>.dream3d` for validation volumes.

Described more in detail in the paper, the entire procedure is a two-step process:

1. The transformer takes an ESBD volume in cubochoric values (the shared model was trained only on volumes with one missing slice) and produces a cubochoric volume of the same shape as the input.

2. The projection algorithm takes the output corresponding to the missing slice and adjacent slices to predict grain IDs for each voxel in the missing slice.


# How to Cite

If you use or build upon this work, please cite the \[ADD PAPER\] that introduced this method:

    Add citation



This work was funded in part by the Data-Driven Discovery of Optimized Multifunctional Material Systems Center of Excellence (D3OM2S CoE), contract number FA8650-19-2-5209.