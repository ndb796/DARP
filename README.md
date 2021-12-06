# DARP: Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning

### Debugging with WandB

* 기본적으로 labeled data의 수가 많은 상황(ratio=2, num_max=1500)에서는 pseudo_orig은 uniform distribution에 가까움
* 하지만 labeled data의 수가 적은 상황(ratio=500, num_max=10)에서는 pseudo_orig은 심한 imbalanced distribution이 됨

<pre>
# Basic running 1 (data_time 속도 빠름)
python3 train.py --gpu 3 --darp --alpha 2 --warm 5 --semi_method fix --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 1 --imb_ratio_u 1 --epoch 500 --val-iteration 500
# Basic running 2 (data_time 속도 느림)
python3 train.py --gpu 3 --darp --alpha 2 --warm 5 --semi_method fix --dataset cifar10 --ratio 500 --num_max 10 --imb_ratio_l 1 --imb_ratio_u 1 --epoch 500 --val-iteration 500
# WandB logging (data_time 속도 느림)
python3 train_with_wandb.py --gpu 3 --darp --alpha 2 --warm 5 --semi_method fix --dataset cifar10 --ratio 500 --num_max 10 --imb_ratio_l 1 --imb_ratio_u 1 --epoch 500 --val-iteration 500
</pre>

<hr>

This repository contains code for the paper
**"Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin.

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `progress`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Scripts
Please check out `run.sh` for the scripts to run the baseline algorithms and ours (DARP).

### Training procedure of DARP 
Train a network with baseline algorithm, e.g., MixMatch on CIFAR-10
```
python train.py --gpu 0 --semi_method mix --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 \
--epoch 500 --val-iteration 500
```
Applying DARP on the baseline algorithm
```
#python train.py --gpu 0 --darp --est --alpha 2 --warm 200 --semi_method mix --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1  \
--epoch 500 --val-iteration 500
```
