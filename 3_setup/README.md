# Code for ``Tabula (Nearly) Rasa''

## Train Language Model

These are the commands used for the random hyperparameter search. Parameters other than those specified will be randomized.


RNN CNLM:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-rnn.py --language german  --save-to wiki-german-nospaces-bptt-rnn-MYID
```

LSTM CNLM:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language english  --save-to wiki-english-nospaces-bptt-MYID --hidden_dim 1024
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language german  --save-to wiki-german-nospaces-bptt-MYID --hidden_dim 1024
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language italian  --save-to wiki-italian-nospaces-bptt-MYID --hidden_dim 1024
```

WNLM:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words.py --language german  --save-to wiki-german-nospaces-bptt-rnn-MYID --hidden_dim 1024 --layer_num 2 
```

Control study: CNLMs with whitespace, with hyperparameters as for the selected CNLMs:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --save-to wiki-english-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language german --layer_num 2 --learning_rate 2.0 --lr_decay 0.7 --save_to wiki-german-nospaces-bptt-WHITESPACE-MYID --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 3 --learning_rate 3.5 --lr_decay 0.98 --save_to wiki-italian-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```


Character-level word  autoencoder baseline:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-encoderbaseline.py --language german --save-to wiki-autoencoder --batchSize 32
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-encoderbaseline.py --language italian --save-to wiki-autoencoder-italian --batchSize 32
```



