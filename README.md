# Neural-IMLS
The offical implentation of [Neural-IMLS](https://arxiv.org/abs/2109.04398)

## Introduction

This is the code for **training neural implicit representations from unoriented noisy 3D point clouds directly**.

It allows to train, test and evaluate the tasks of surface reconstruction. 

We provide the code for training a new model and test that model on your own data. Besides, we provie some reconstruction results of our method.


![insight](assets/insight.pdf)

### Requirements

Our codebase uses [PyTorch](https://pytorch.org/).

The code was tested with Python 3.7.9, torch 1.11.0, CUDA 11.3 on Ubuntu 18.04 (should work with later versions).

The most important part of environment is [pytorch3d](https://github.com/facebookresearch/pytorch3d)

```python
conda env create -f imls.yml
```

### Training

To train model to overfit one shape, run this command:

```python
python train.py --pts_dir <path_to_data> --name <experiment_name>

For example:
python train.py --pts_dir data/ori_bunny.xyz.npy --name famous_ori_bunny --patch_radius 0.03 --points_per_patch_max 100
```

### Test and Evaluation

To evaluate model, run:

```python
python test.py --model_path <path_to_trained_model> --name <experiment_name> --mesh_path <path_to_gt_mesh>

For example:
python test.py --model_path experiment/famous_noisefree_Armadillo/epoch_35.pth --name famous_noisefree_Armadillo --mesh_path mesh/Armadillo.obj
```

### Results

Our surface reconstruction results (meshes) on SRB (The Surface Reconstruction Benchmark from [DGP](https://github.com/fwilliams/deep-geometric-prior)) are available for download [here]().

Our surface reconstruction results (meshes) on ABC no.n, ABC var.n, FAMOUS no.n, FAMOUS med.n (data preprocessed by [Points2Surf](https://github.com/ErlerPhilipp/points2surf)) are available for download [here]().

### Citation

If you use our work, please cite our paper:

```bibtex
@misc{2109.04398,
    Author = {Zixiong Wang and Pengfei Wang and Qiujie Dong and Junjie Gao and Shuangmin Chen and Shiqing Xin and Changhe Tu and Wenping Wang},
    Title = {Neural-IMLS: Learning Implicit Moving Least-Squares for Surface Reconstruction from Point Clouds},
    Year = {2022},
    Eprint = {arXiv:2109.04398},
}
```

