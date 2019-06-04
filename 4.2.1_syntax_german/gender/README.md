# Code for ``Tabula (Nearly) Rasa''

## Morphosyntax

### Gender

```
LSTM CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

RNN CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-RNN.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083

WNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
```


restrict to Word LSTM vocabulary:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-LEXICON.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-RNN-LEXICON.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083
```



