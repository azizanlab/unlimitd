# UNCERTAINTY-AWARE META-LEARNING IN MULTIMODAL TASK DISTRIBUTIONS

This is the official code for "Uncertainty-Aware Meta-Learning in Multimodal Task Distributions", by César Almecija, Apoorva Sharma and Navid Azizan.

[[Arxiv](https://arxiv.org/abs/2210.01881)]

## Dependencies, datasets and trained models

This repository uses [`jax`](https://jax.readthedocs.io/en/latest/), in addition to usual machine-learning libraries.
See [this link](https://github.com/google/jax#installation) for a guide to install `jax`.

The vision problem dataset (Shapenet1D), from [Gao et al.](https://arxiv.org/abs/2203.04905) is available at [this page](https://github.com/boschresearch/what-matters-for-meta-learning/blob/main/data/ShapeNet1D.tar.xz).
Extract the file and then paste the folder path into `dataset_shapenet1d.py` (function `load_shapenet1d`).

If you wish to skip the training process prior to evaluation, you can use our trained models, available at [this page](https://drive.google.com/drive/folders/1zEDs32yC5YfxVtFCz4WcnbU8qFhizON-?usp=sharing), and directly skip to the evaluation section.
Simply extract the content of the file into `/logs_final/`.

Note: across all the files, `maddox_noise` represents assumed noise in the linear regression, refering to  [Maddox et al.](proceedings.mlr.press/v130/maddox21a/maddox21a.pdf)'s idea (transfer learning with Bayesian inference on a linearized model).

## Training with `UNLIMTD`
We provide some notebooks / files to retrain the models on your end.

### Training on the simple regression problems

Unimodal cases (sines task dataset):
* `UNLIMTD-I` (infinite dataset): run the notebook `unlimtd_i_uni_modal_infinite.ipynb`
* `UNLIMTD-I` (finite dataset): run the notebook `unlimtd_i_uni_modal_finite.ipynb`
* `UNLIMTD-R` (infinite dataset): run the notebook `unlimtd_r_uni_modal_infinite.ipynb`
* `UNLIMTD-R` (finite dataset): run the notebook `unlimtd_r_uni_modal_finite.ipynb`
* `UNLIMTD-F` (infinite dataset): run the notebook `unlimtd_f_uni_modal_infinite.ipynb`
* `UNLIMTD-F` (finite dataset): run the notebook `unlimtd_f_uni_modal_finite.ipynb`

Multimodal cases (sines+lines task dataset):
* `UNLIMDT-F` (mixture of GPs): run the notebook `unlimtd_f_multi_modal_mixture.ipynb`
* `UNLIMDT-F` (single GP): run the notebook `unlimtd_f_multi_modal_singGP.ipynb`

### Training on the vision problem

This problem is unimodal on the Shapenet1D dataset:
* `UNLIMTD-I`: run the file `vision_unlimtd_i.py`
* `UNLIMTD-R`: run the file `vision_unlimtd_r.py`
* `UNLIMTD-F`: run the files `vision_unlimtd_f_before.py`, then `vision_unlimtd_f_fim.py` and finally `vision_unlimtd_f_after.py` (we split the different parts of the meta-training so that the GPU does not run out of RAM)

### Training the baselines

* Unimodal case (sines task dataset), `MAML`: run the notebook `maml_uni_modal.ipynb`.
* Multimodal case (sines+lines task dataset), `MAML`: run the notebook `maml_multi_modal.ipynb`
* Multimodal case (sines+lines task dataset), `MMAML`: run [MMAML with FiLM](https://github.com/vuoristo/MMAML-Regression#film). Make sure to:
  * change `simple_functions.generate_sinusoid_batch` to a POSITIVE phase
  * change the `bias` to 1 in `simple_functions.MixedFunctionsMetaDataset`.
  * add the following arguments when running the training: `--slope-range -1.0 1.0 --intersect-range 0.0 0.0` (to have the same lines task dataset than UNLIMTD)

* Vision problem (uni-modal Shapenet1D), `MAML`: run the file `vision_maml.py`.

## Evaluation
If you wish to evaluate the models, we provide some notebooks to build the same plots as the ones presented in the paper.

### Evaluation of the simple regression problems
* Unimodal case (sines task dataset): run the notebook `plots_uni_modal.ipynb`.
* Multimodal case (sines+lines task dataset): run the notebook `plots_multi_modal.ipynb`.
Note that the results of MMAML are hardcoded inside. If you wish to recompute them, you'll need to run the following, after training MMAML (MMAML is not included in our trained models). If the training directory is `train_dir/2mods-mmaml-5steps-10K/`, `specify_checkpoint` is the name of the checkpoint, `K` is the number of context inputs (between 1 and 10) and `L` the number of query inputs (100): `python3 ~/MMAML/MMAML-Regression/main.py --dataset mixed --num-batches 100 --model-type gated --fast-lr 0.001 --meta-batch-size 50 --num-samples-per-class L + K --num-val-samples L --noise-std 0.05 --hidden-sizes 100 100 100 --device cuda --num-updates 5 --output-folder 2mods-mmaml-5steps-10K --bias-transformation-size 20 --disable-norm --embedding-type LSTM --embedding-dims 200  --inner-loop-grad-clip 10 --slope-range -1.0 1.0 --intersect-range 0.0 0.0 --eval --checkpoint train_dir/2mods-mmaml-5steps-10K/specify_checkpoint`

### Evaluation of the vision problem
* Vision problem (unimodal): run the notebook `plots_vision.ipynb` (also has example of predictions)
