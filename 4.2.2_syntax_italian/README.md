# Code for ``Tabula (Nearly) Rasa''


## Italian Morphosyntactic Tests
All three tests are done in one go by the following commands:
```
LSTM CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024  --layer_num 3 --learning_rate 2.0 --load-from wiki-italian-nospaces-bptt-887669069 --sequence_length 50 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

RNN CNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj_RNN.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048  --layer_num 2 --learning_rate 2.0 --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 50 --weight_dropout_hidden 0.0 --weight_dropout_in 0.01

WNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj_WORDS.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 300 --char_noise_prob 0.0 --hidden_dim 1024  --layer_num 2 --learning_rate 2.0 --load-from wiki-german-nospaces-bptt-words-20176990 --sequence_length 50 --weight_dropout_hidden 0.0 --weight_dropout_in 0.01
```



