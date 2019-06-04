# Code for ``Tabula (Nearly) Rasa''

## POS classification
LSTM CNLM:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```

RNN CNLM:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS-rnn.py --batchSize 256 --char_dropout_prob 0.01 --char_embedding_size 50 --char_noise_prob 0.0 --hidden_dim 2048 --language german --layer_num 2 --learning_rate 0.1 --lr_decay 0.95 --nonlinearity tanh --load-from wiki-german-nospaces-bptt-rnn-237671415 --sequence_length 30 --verbose True --weight_dropout_hidden 0.0 --weight_dropout_in 0.0 --train_size 20

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS-rnn.py --batchSize 256 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language italian --layer_num 2 --learning_rate 0.004 --lr_decay 0.98 --nonlinearity tanh --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 20 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --train_size 20
```

WNLM:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 1.2 --lr_decay 0.98 --load-from wiki-italian-nospaces-bptt-words-316412710 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```

WNLM, including OOVs:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings-withOOV.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings-withOOV.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 1.2 --lr_decay 0.98 --load-from wiki-italian-nospaces-bptt-words-316412710 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```

Autoencoder:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language italian --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder-italian
```


