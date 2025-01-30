import os
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--fold", type=int, default=10)
parser.add_argument("--label_frac", type=float, default=1.00)
parser.add_argument("--pretrain", type=str, default='Res50_on_ImageNet')
parser.add_argument("--dataset", type=str, default='Camelyon16')
parser.add_argument("--use_random_inst", type=str, default='False')
parser.add_argument("--init_type", type=str, default='normal')
parser.add_argument("--mask_ratio", type=float, default=0.1)
parser.add_argument("--degree", type=int, default=0)
parser.add_argument("--model", type=str, default='v1')
parser.add_argument("--pretrain_dir", type=str, default='Null')
parser.add_argument("--lwc", type=str, default='Null')
parser.add_argument("--distill_loss", type=str, default='Null')
args = parser.parse_args()


#######################################################
# train and val
#######################################################
for k in range(10):
    cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase train --dataset {} --pretrain {} --model {} --mask_ratio {} \
     --fold {} --label_frac {} --lr {} --k {} --degree {} --init_type {} --pretrain_dir {} --lwc {} --distill_loss {} --use_random_inst {}'.format(args.gpu_id, args.dataset, args.pretrain, args.model, args.mask_ratio, args.fold, args.label_frac, args.lr, k, args.degree, args.init_type, args.pretrain_dir, args.lwc, args.distill_loss, args.use_random_inst)
    os.system(cmd)
    cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase test --dataset {} --pretrain {} --model {} --mask_ratio {} \
     --fold {} --label_frac {} --lr {} --k {} --degree {} --init_type {} --pretrain_dir {} --lwc {} --distill_loss {} --use_random_inst {}'.format(args.gpu_id, args.dataset, args.pretrain, args.model, args.mask_ratio, args.fold, args.label_frac, args.lr, k, args.degree, args.init_type, args.pretrain_dir, args.lwc, args.distill_loss, args.use_random_inst)
    os.system(cmd)