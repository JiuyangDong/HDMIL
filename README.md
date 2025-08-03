# HDMIL: Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning [CVPR2025]
The official implementation of "Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning".

🚨 **Notice: Updated Version Available**
This repository corresponds to our earlier work published at **CVPR 2025**.  
We have released a significantly extended and improved version in:
"AHDMIL: Asymmetric Hierarchical Distillation Multi-Instance Learning for Fast and Accurate Whole-Slide Image Classiffcation"  
💻 [Access the new GitHub repository](https://github.com/JiuyangDong/AHDMIL)

We strongly recommend referring to the updated version for improved performance, additional experiments, and latest implementations.


## How to use: training & validation & testing scripts
 
### For the Camelyon16 dataset
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 0 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.6 
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/C10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.6/ckpts/ \
    --mask_ratio 0.6  --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False 
```

### For the TCGA-NSCLC dataset
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-NSCLC --gpu_id 0 --lr 3e-5 --fold 10 \
    --label_frac 1.00 --degree 16 --init_type xaiver --model v4 --mask_ratio 0.7  
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-NSCLC --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/N10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=16/lr=3e-05_maskratio=0.7/ckpts/ \
    --mask_ratio 0.7  --degree 16 --lwc mbv4t --distill_loss l1 --use_random_inst False 
```

### For the TCGA-BRCA dataset
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-BRCA --gpu_id 0 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.7  
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-BRCA --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/B10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.7/ckpts/ \
    --mask_ratio 0.7  --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False 
```

### For the TCGA-RCC dataset
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-RCC --gpu_id 0 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.6
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-RCC --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/R10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.6/ckpts/ \
    --mask_ratio 0.6 --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False
```
## Reference
If you found our work useful in your research, please consider citing our works(s) at:
```
@inproceedings{dong2025fast,
  title={Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning},
  author={Dong, Jiuyang and Jiang, Junjun and Jiang, Kui and Li, Jiahan and Zhang, Yongbing},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30818--30828},
  year={2025}
}
```


© 2025 Dong Jiuyang. This code is released under the GPLv3 license and is intended for non-commercial academic research only.

