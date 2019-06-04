# Code for ``Tabula (Nearly) Rasa''


## Segmentation

### Find Predictive Neuron

```
python detectBoundariesUnit_Hidden_NoWhitespace.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python detectBoundariesUnit_Hidden_NoWhitespace.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language german --layer_num 2 --learning_rate 2.0 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01 --load-from wiki-german-nospaces-bptt-910515909

python detectBoundariesUnit_Hidden_NoWhitespace.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 3.5  --weight_dropout_hidden 0.05 --weight_dropout_in 0.0 --load-from wiki-italian-nospaces-bptt-855947412
```


### Running Classifier on Controlled Input

```
python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_Test.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_Test.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language german --layer_num 2 --learning_rate 2.0 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01 --load-from wiki-german-nospaces-bptt-910515909

python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_Test.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 3.5  --weight_dropout_hidden 0.05 --weight_dropout_in 0.0 --load-from wiki-italian-nospaces-bptt-855947412

```


### Running Classifier on Running Text

```
python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_RealText.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_RealText.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language german --layer_num 2 --learning_rate 2.0 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01 --load-from wiki-german-nospaces-bptt-910515909

python detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_RealText.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 3.5  --weight_dropout_hidden 0.05 --weight_dropout_in 0.0 --load-from wiki-italian-nospaces-bptt-855947412
```

Use `detectBoundariesUnit_Hidden_ExtractPattern_NoWhitespace_Classifier_RealText_FullClassifier.py` for the classifier based on the full hidden state.



