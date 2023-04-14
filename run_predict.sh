#!/usr/bin/env bash
python transformer_classificatiom_predict.py \
--test_path data/test_binary.csv \
--hetiones_path data/df_triples.csv \
--config_file result/args.json \
--batch_size 1024 \
--parent_model polo_model \
--save_dir output/predict