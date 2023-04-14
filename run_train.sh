#!/usr/bin/env bash
python transformer_classification.py \
--train_path data/train_binary.csv \
--test_path data/test_binary.csv \
--entities_path data/entities_df.csv \
--relations_path data/kg_relations.txt \
--save_dir output \
--batch_size 1024 \
--epochs 20 \
--max_len 9 \
--nhead 5 \
--d_model 100 \
--output_dim 2