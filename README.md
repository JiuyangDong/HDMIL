# HDMIL
The official implementation of "Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning"

## training & validation & testing
### for Binary Classification Tasks
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 1 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.1  
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/C10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.1/ckpts/ \
    --mask_ratio 0.1  --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False 
```
