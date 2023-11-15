#!/usr/bin/env bash

config2=$1
config1=../configs/Eval/llff/$1
config2=../configs/Eval/nerf_synthetic/$1
config3=../configs/Eval/deepvoxels/$1

python eval.py --config $config1 --eval_scenes horns 
python eval.py --config $config1  --eval_scenes trex 
python eval.py --config $config1  --eval_scenes room 
python eval.py --config $config1 --eval_scenes flower 
python eval.py --config $config1 --eval_scenes orchids 
python eval.py --config $config1 --eval_scenes leaves 
python eval.py --config $config1 --eval_scenes fern 
python eval.py --config $config1 --eval_scenes fortress 

python eval.py --config $config3 --eval_scenes cube 
python eval.py --config $config3 --eval_scenes vase 
python eval.py --config $config3 --eval_scenes greek
python eval.py --config $config3 --eval_scenes armchair 

# python eval.py --config $config2 --eval_scenes mic 
# python eval.py --config $config2  --eval_scenes chair 
# python eval.py --config $config2  --eval_scenes lego 
# python eval.py --config $config2 --eval_scenes ficus 
# python eval.py --config $config2 --eval_scenes materials 
# python eval.py --config $config2 --eval_scenes hotdog 
# python eval.py --config $config2 --eval_scenes ship 
# python eval.py --config $config2 --eval_scenes drums 
