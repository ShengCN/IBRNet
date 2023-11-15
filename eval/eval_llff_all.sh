#!/usr/bin/env bash

config_file=$1

python eval.py --config $config_file --eval_scenes horns 
python eval.py --config $config_file  --eval_scenes trex 
python eval.py --config $config_file  --eval_scenes room 
python eval.py --config $config_file --eval_scenes flower 
python eval.py --config $config_file --eval_scenes orchids 
python eval.py --config $config_file --eval_scenes leaves 
python eval.py --config $config_file --eval_scenes fern 
python eval.py --config $config_file --eval_scenes fortress 

