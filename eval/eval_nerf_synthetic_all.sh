#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes mic &
# CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes chair &
# CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes lego &
# CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes ficus &
# CUDA_VISIBLE_DEVICES=4 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes materials &
# CUDA_VISIBLE_DEVICES=5 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes hotdog &
# CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes ship &
# CUDA_VISIBLE_DEVICES=7 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes drums &


config_file=$1
python eval.py --config $config_file --eval_scenes mic 
python eval.py --config $config_file  --eval_scenes chair 
python eval.py --config $config_file  --eval_scenes lego 
python eval.py --config $config_file --eval_scenes ficus 
python eval.py --config $config_file --eval_scenes materials 
python eval.py --config $config_file --eval_scenes hotdog 
python eval.py --config $config_file --eval_scenes ship 
python eval.py --config $config_file --eval_scenes drums 

