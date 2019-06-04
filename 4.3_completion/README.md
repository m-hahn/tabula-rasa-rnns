# Code for ``Tabula (Nearly) Rasa''




## MSR Sentence Completion


Run Wikipedia CNLM/WNLM models on the completion task:
```
python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from wiki-english-nospaces-bptt-words-805035971 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 

python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01

```

Post-Train Wikipedia models on the in-domain training set:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from wiki-english-nospaces-bptt-words-805035971 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --save-to holmes-words-from-805035971-MYID

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.5 --lr_decay 0.95  --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0 --save-to holmes-from-282506230-MYID

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --save-to holmes-rnn-from-891035072-MYID --lr_decay 0.9 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

Run the resulting models:
```
python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-805035971-218115572 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-282506230-684739660

python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-282506230-932742390

python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from holmes-rnn-from-891035072-134123184 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01

```


Train fresh models on in-domain training set:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --save-to holmes-words-from-fresh-MYID

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --save-to holmes-rnn-from-fresh-MYID --lr_decay 0.9 --nonlinearity relu --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

Run the resulting models:
```
python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-fresh-77128193 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-fresh-754593543

python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from holmes-rnn-from-fresh-388239891 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

Train a WNLM model with fresh vocabulary:
```
python char-lm-ud-stationary-completion-words-VOCAB.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-fresh-vocab-237230358 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0
```




