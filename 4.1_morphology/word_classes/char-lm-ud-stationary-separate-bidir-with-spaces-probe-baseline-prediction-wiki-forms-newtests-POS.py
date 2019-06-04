from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
# POS induction: logistic classifier operating on hidden states


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=50)
parser.add_argument("--train_size", type=int, default=20)



args=parser.parse_args()
print(args)





import corpusIteratorWiki


def plusL(its):
  for it in its:
       for x in it:
           yield x

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

# Open the character vocabulary
try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError: # Create it from scratch if it doesn't exist yet
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary

    with open(WIKIPEDIA_HOME+"/"+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop

# Create the language model
rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]

rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

#optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings} #, "optim" : optim}

print("Loading model")
if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      print(checkpoint[name].keys())
      module.load_state_dict(checkpoint[name])
else:
   assert False



####################################


from torch.autograd import Variable



# Do not do dropout here (no training will happem)
rnn_drop.train(False)

lossModule = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)

# Takes a string and transforms into a list of integers
def encodeWord(word):
      numeric = []
      for char in word:
           numeric.append(stoi[char]+3 if char in stoi else 2) # 2 is the code for OOV
      return numeric



# Input: a list of words
# Output: a list of hidden LSTM states as Numpy arrays
def encodeListOfWords(words):
    numeric = [encodeWord(word) for word in words] # transforms each word into a list of integers
    maxLength = max([len(x) for x in numeric])
    for i in range(len(numeric)): # pad at the beginning, if necessary, so that all lists have the same length
       numeric[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False) # create a tensor from the integer lists
    
    input_cut = input_tensor_forward
    embedded_forward = char_embeddings(input_cut) # run through the embedding layer
    out_forward, hidden_forward = rnn_drop(embedded_forward, None) # run the LSTM
    hidden = hidden_forward[0].data.cpu().numpy() # obtain the final hidden state of the LSTM
    return [hidden[0][i] for i in range(len(words))] # return, for each input word, the corresponding final LSTM state


import numpy as np


# paths of vocabularies with POS annotation
vocabPath = {"german" : WIKIPEDIA_HOME+"german-wiki-word-vocab-POS.txt", "italian" : WIKIPEDIA_HOME+"italian-wiki-word-vocab-POS.txt"}[args.language]


# language-specific rules for detecting verbs and nouns from the POS annotation
def detectVerb(pos):
  if args.language == "german":
      return pos.startswith("v")
  elif args.language == "italian":
      return pos.startswith("ver") 
def detectNoun(pos):
   if args.language == "german":
      return pos.startswith("n")
   elif args.language == "italian":
       return pos == "noun"

# Select all nouns and verbs from the POS-annotated vocabularies
nouns = []
verbs = []
nounsTotal = set()
verbsTotal = set()
incorporating = True
with open(vocabPath, "r") as inFile:
  for line in inFile:
    line = line.strip().split("\t")
    pos = line[1]
    if detectVerb(pos):
        verbsTotal.add(line[0])
        if incorporating:
           verbs.append(line[0])
    elif detectNoun(pos):
        nounsTotal.add(line[0])
        if incorporating:
            nouns.append(line[0])
    if incorporating and len(line[2]) <= 3 and int(line[2]) < 10:
        incorporating = False

# Restrict to words that only occur with one of the two POS labels
nouns = [x for x in nouns if x not in verbsTotal]
verbs = [x for x in verbs if x not in nounsTotal]
print(len(nouns))
print(len(verbs))

# Here we define which words we consider. In the German and Italian experiments, we restricted to words with a fixed ending to rule out the most basic morphological cues.
def criterion(word):
    if args.language == "german":
       return word.endswith("en")
    elif args.language == "italian":
       return word.endswith("re")

nounsInN = [x for x in nouns if criterion(x)]
verbsInN = [x for x in verbs if criterion(x)]

# How many words do we get?
print(len(nounsInN))
print(len(verbsInN))

print(nounsInN[:100])
print(verbsInN[:100])

# How many words we consider in each run of the experiment (This is set to 500 to make each run faster. With hindsight, an alternative would have been to precompute the LSTM encodings and reuse them in every step.)
sampleSize = 500

accuracies = []
for _ in range(100): # do 100 runs of the experiment

  # randomly select a subset
  random.shuffle(nounsInN)
  random.shuffle(verbsInN)
  nounsInNSelected = nounsInN[:sampleSize]# len(verbsInN)]
  verbsInNSelected = verbsInN[:sampleSize] #len(verbsInN)]
  
  # collect the hidden states for all selected words
  encodedNounsInN = encodeListOfWords([x for x in nounsInNSelected])
  encodedVerbsInN = encodeListOfWords([x for x in verbsInNSelected])
  
  # predictor and dependent variable for logistic regression
  predictors = encodedNounsInN + encodedVerbsInN
  dependent = [0 for _ in encodedNounsInN] + [1 for _ in encodedVerbsInN]
  
  # split into training and test partition
  from sklearn.model_selection import train_test_split
  x_train_nouns, x_test_nouns, y_train_nouns, y_test_nouns = train_test_split(encodedNounsInN, [0 for _ in encodedNounsInN], test_size=1-args.train_size/sampleSize, shuffle=True)
  x_train_verbs, x_test_verbs, y_train_verbs, y_test_verbs = train_test_split(encodedVerbsInN, [1 for _ in encodedVerbsInN], test_size=1-args.train_size/sampleSize, shuffle=True)

  x_train = x_train_nouns + x_train_verbs
  y_train = y_train_nouns + y_train_verbs
  x_test = x_test_nouns + x_test_verbs
  y_test = y_test_nouns + y_test_verbs


  # Train the logistic regression
  from sklearn.linear_model import LogisticRegression
  print("regression")
  logisticRegr = LogisticRegression()
  logisticRegr.fit(x_train, y_train)
  score = logisticRegr.score(x_test, y_test)
  print(score)
  accuracies.append(score)

# Compute average, standard deviation, and 0.05/0.95 quantiles of accuracy.
print("--")

print(sum(accuracies)/100)

meanAccuracy = sum(accuracies)/100
meanSquaredAccuracy = sum([x**2 for x in accuracies])/100
import math
standardDeviation = math.sqrt(meanSquaredAccuracy - meanAccuracy**2)

print(standardDeviation)


accuracies = sorted(accuracies)

quantile_lower = accuracies[int(0.05 * 100)]
quantile_upper = accuracies[int(0.95 * 100)]

print((quantile_lower, quantile_upper))

with open(f"/checkpoint/mhahn/pos/{__file__}_"+args.language+"_"+str(args.train_size)+"_"+args.load_from, "w") as outFile:
   print(args.language, file=outFile)
   print(args.train_size, file=outFile)
   print(args.load_from, file=outFile)
   print(sum(accuracies)/100, file=outFile)  
   print(standardDeviation)
   print(quantile_lower)
   print(quantile_upper)


