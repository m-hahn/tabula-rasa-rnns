#!/bin/bash


#SBATCH --job-name=it_search_wiki
#SBATCH --output=/checkpoint/mhahn/jobs/search-english-wiki-%j.out
#SBATCH --error=/checkpoint/mhahn/jobs/search-english-wiki-%j.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --mem=7G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=4320

module purge

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 3 --learning_rate 3.5 --lr_decay 0.98 --save-to wiki-italian-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0


