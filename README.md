# 3D Myotube Synthesis

A pipeline to generate synthetic 3D myotube datasets 

## Overview

This repository implements the methods described in our preprint [Data Synthesis Improves 3D Myotube Instance Segmentation](https://arxiv.org/abs/2604.14720).

Key features:
- Simulation of 3D myotube annotations.
- Simulation of imaging effects and artifacts.
- SSL pretraining, retraining, and inference of the simple models demonstrated in our article.

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Simulate annotations

Generate an annotation dataset using the synth.py script.

Tune the parameters according to your imaging setup, so the myotube size fits the zoom factor, image resolution and dimensions are the same as your real image, etc.

The parameters used in our experiments are the script defaults. A default run can be started with:

```bash
python synth.py
```

Input parameter values by adding 
```bash
python synth.py --parameter_name parameter_value

#example
python synth.py --num-images 30 --shape 1024,1024,128 --num-myos 64
```
to the call. All available parameters are listed below:
| Parameter | Default | Description|
| -------------------------------- | --------------: | -------------------------------------------------------------------------- |
| `--output-folder`                |   `annotations` | Target folder for generated `.tif` files.                                  |
| `--output-stem`                  |          `syn_` | Filename prefix for generated images.                                      |
| `--num-images`, `--num-pictures` |            `15` | Number of synthetic volumes to generate.                                   |
| `--num-polys`, `--num-myos`      |            `64` | Number of myotubes/myocytes to place per image.                            |
| `--max-tries`                    |             `5` | Maximum number of failed placement attempts before moving on.              |
| `--place-poly-max-tries`         |            `20` | Maximum attempts to place one myotube.                                     |
| `--shape`                        | `1024,1024,128` | Output volume shape as `H,W,Z`.                                            |
| `--min-thickness`                |             `8` | Minimum XY thickness of a myotube.                                         |
| `--max-thickness`                |            `30` | Maximum XY thickness of a myotube.                                         |
| `--length-min`                   |           `400` | Minimum myotube centerline length.                                         |
| `--length-max`                   |          `1025` | Maximum myotube centerline length, exclusive.                              |
| `--branch-prob`                  |          `0.15` | Probability that a myotube receives a secondary branch.                    |
| `--branch-min-len`               |            `40` | Minimum secondary branch length.                                           |
| `--branch-max-len`               |           `220` | Maximum secondary branch length.                                           |
| `--branch-min-end-dist`          |            `10` | Minimum distance of the secondary branch endpoint from the trunk endpoint. |
| `--branch-max-end-dist`          |           `100` | Maximum distance of the secondary branch endpoint from the trunk endpoint. |
| `--wiggle-amp`                   |         `150.0` | Scale factor for XY centerline wiggle amplitude.                           |
| `--z-wiggle-scale`               |           `0.3` | Scale factor for Z wiggle amplitude relative to XY wiggle.                 |
| `--xy-degree`                    |            `10` | Polynomial degree used for XY centerline generation.                       |
| `--xy-bound`                     |          `0.75` | Bound for XY centerline sampling before scaling.                           |
| `--xy-max-wiggle`                |         `0.095` | Maximum allowed mean squared wiggle of the XY centerline.                  |
| `--xy-coeff-damper`              |          `1.35` | Dampening factor for higher-order XY polynomial coefficients.              |
| `--z-degree`                     |             `6` | Polynomial degree used for Z centerline generation.                        |
| `--z-bound`                      |          `0.25` | Bound for Z centerline sampling before scaling.                            |
| `--z-max-wiggle`                 |         `0.005` | Maximum allowed mean squared wiggle of the Z centerline.                   |
| `--z-coeff-damper`               |          `1.35` | Dampening factor for higher-order Z polynomial coefficients.               |
| `--xy-straight-prob`             |           `0.4` | Probability of inserting a straight segment into the XY centerline.        |
| `--xy-straight-max-length`       |           `500` | Maximum length of an inserted straight XY segment.                         |
| `--z-straight-prob`              |           `0.5` | Probability of inserting a straight segment into the Z centerline.         |
| `--z-straight-max-length`        |            `50` | Maximum length of an inserted straight Z segment.                          |
| `--xy-max-wiggles`               |           `5.0` | Maximum number of sine periods for XY thickness variation.                 |
| `--xy-sin-influence`             |          `0.15` | Strength of sinusoidal variation in XY thickness.                          |
| `--z-max-wiggles`                |           `5.0` | Maximum number of sine periods for Z thickness variation.                  |
| `--z-sin-influence`              |          `0.15` | Strength of sinusoidal variation in Z thickness.                           |
| `--z-mean-thickness-min`         |             `5` | Minimum mean Z thickness.                                                  |
| `--z-mean-thickness-max`         |            `15` | Maximum mean Z thickness.                                                  |
| `--z-max-thickness-extra-min`    |             `3` | Minimum additional Z thickness above the mean.                             |
| `--z-max-thickness-extra-max`    |             `5` | Maximum additional Z thickness above the mean.                             |
| `--xy-local-max-thickness-min`   |             `3` | Minimum additional XY thickness above the mean.                            |
| `--xy-local-max-thickness-max`   |             `8` | Maximum additional XY thickness above the mean.                            |


### 3. Simulate effects

Generate a training dataset using the effects.py script.

Specify the input and output folder, as well as filename stems with four parameters:

| Parameter | Default | Description|
| -------------------------------- | --------------:  | -------------------------------------------------------------------------- |
| `----output-folder`                |   `syn`        | Target folder for generated `.tif` files.                                  |
| `----output-stem`                |   `syn_effects_` | Filename stem for generated `.tif` files.                                  |
| `----input-folder`                |   `annotations` | Input folder of previously generated annotations.                                  |
| `----input-stem`                |   `syn_`          | Filename stem of previously generated annotations.                                  |

### 4. (Optional) Training 

We demonstrate the data synthesis pipeline by training segmentation models on the generated data, as well as a CycleGAN adapted set of the same data.
The models, losses, and training routines are by no means optimally designed, altough we plan to tackle this next.
Nevertheless, the scripts we used are in this repo, use
```bash
python train_* --help 
```
to see the available parameters.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{exler2026data,
  title={Data Synthesis Improves 3D Myotube Instance Segmentation},
  author={Exler, David and Friederich, Nils and Kr{\"u}ger, Martin and Jbeily, John and Vitacolonna, Mario and Rudolf, R{\"u}diger and Mikut, Ralf and Reischl, Markus},
  journal={arXiv preprint arXiv:2604.14720},
  year={2026}
}
```