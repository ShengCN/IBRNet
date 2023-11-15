#!/usr/bin/env bash
config_file=$1

python eval.py --config $config_file --eval_scenes cube 
python eval.py --config $config_file --eval_scenes vase 
python eval.py --config $config_file --eval_scenes greek
python eval.py --config $config_file --eval_scenes armchair 