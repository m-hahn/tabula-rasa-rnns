# Code for ``Tabula (Nearly) Rasa''

## Morphosyntax

### Case agreement

```
LSTM CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

WNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

RNN CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-RNN.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083

Restrict the CNLM to the vocabulary of the WNLM (control study, not reported):
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-LEXICON.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

Same for RNN CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-RNN-LEXICON.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083
```


