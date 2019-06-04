# This script does the German plural experiments based on the stimulus files


# python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-fromfile.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME

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

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
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


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


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

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings} #, "optim" : optim}

print("Loading model")
if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      print(checkpoint[name].keys())
      module.load_state_dict(checkpoint[name])
#else:
#   assert False
####################################





from torch.autograd import Variable


rnn_drop.train(False)




def encodeWord(word):
      numeric = [[]]
      for char in word:
           numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric



def encodeListOfWordsOld(words):
    numeric = [encodeWord(word)[0] for word in words]
    maxLength = max([len(x) for x in numeric])
    for i in range(len(numeric)):
       numeric[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
    
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)
    hidden = hidden_forward[0].data.cpu().numpy()
    return [hidden[0][i] for i in range(len(words))]



def encodeListOfWords(words):
    numeric = [encodeWord(word)[0] for word in words]
    lengths = [len(x) for x in numeric]
    maxLength = max(lengths)

    for i in range(len(numeric)):
       numeric[i] = numeric[i] + ([0]*(maxLength-len(numeric[i])))
    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
    
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]
    embedded_forward = char_embeddings(input_cut)
    
    result = [None for _ in words]

    lengthsToWords = {}
    for i in range(len(lengths)):
        lengthsToWords[lengths[i]] = lengthsToWords.get(lengths[i], []) + [i]

    hidden_forward = None
    for i in range(1,maxLength+1):
       out_forward, hidden_forward = rnn_drop(embedded_forward[i-1].unsqueeze(0), hidden_forward)
       if i in lengthsToWords:
          for j in lengthsToWords[i]:
#               print(j, len(result), hidden_forward[0].size())
               result[j] = hidden_forward[0][:,j].flatten().data.cpu().numpy()

    return result


import numpy as np


formations = {"e" : set(), "n" : set(), "s" : set(), "same" : set(), "r" : set()}


for group in formations:
  with open(f"stimuli/german-plurals-{group}.txt", "r") as inFile:
     formations[group] = [tuple(x.split(" ")) for x in inFile.read().strip().split("\n")]
     print(len(formations[group]))


print(formations["n"])
print(formations["same"])


# classify singulars vs plurals
print("trained on n, s, e")



encodedPluralsTotal = [] #encodeListOfWords(["."+y for y in plurals])
encodedSingularsTotal = [] #encodeListOfWords(["."+x for x in singulars])

def redprod1(stimuli):
     encodedPluralsTotalD = dict(encodedPluralsTotal)
     encodedSingularsTotalD = dict(encodedSingularsTotal)

     encodedPluralsR = [encodedPluralsTotalD[y] for x, y in stimuli]
     encodedSingularsR = [encodedSingularsTotalD[x] for x, y in stimuli]
    
     encodedPlurals2 =   encodeListOfWords(["."+y for x, y in stimuli])
     encodedSingulars2 =   encodeListOfWords(["."+x for x, y in stimuli])

     encodedPlurals3 =   encodeListOfWords(["."+y for x, y in stimuli])
     encodedSingulars3 =   encodeListOfWords(["."+x for x, y in stimuli])


     for i in range(len(stimuli)):

   #      print(stimuli[i])
   #      print(encodedPluralsR[i] - encodedPlurals2[i])
   #      print(encodedSingularsR[i] - encodedSingulars2[i])
   #      print(encodedSingularsR[i] - encodedSingulars3[i])

         if max(encodedSingularsR[i] - encodedSingulars2[i]) > 0.00000001:
            assert False, stimuli[i]




def redprod2(stimuli):

     encodedPluralsR = [encodedPluralsTotal[y] for x, y in stimuli]
     encodedSingularsR = [encodedSingularsTotal[x] for x, y in stimuli]
    
     encodedPlurals2 =   encodeListOfWords(["."+y for x, y in stimuli])
     encodedSingulars2 =   encodeListOfWords(["."+x for x, y in stimuli])

     encodedPlurals3 =   encodeListOfWords(["."+y for x, y in stimuli])
     encodedSingulars3 =   encodeListOfWords(["."+x for x, y in stimuli])


     for i in range(len(stimuli)):

#         print(stimuli[i])
#         print(encodedPluralsR[i] - encodedPlurals2[i])
#         print(encodedSingularsR[i] - encodedSingulars2[i])
#         print(encodedSingularsR[i] - encodedSingulars3[i])

         if max(encodedSingularsR[i] - encodedSingulars2[i]) > 0.00000001:
            assert False, stimuli[i]


hasVisitedS = False

print("Encoding...")
 
for group, pairs in formations.items():
  print(group)
  pairs = list(pairs)
  sings = [x[0] for x in pairs]
  plurs = [x[1] for x in pairs]
  for pair in pairs:
         assert pair[0] not in encodedSingularsTotal
         assert pair[1] not in encodedPluralsTotal
  encodedSingularsTotal += zip(sings, encodeListOfWords(["."+x+"." for x in sings]))
  encodedPluralsTotal += zip(plurs, encodeListOfWords(["."+x+"." for x in plurs]))
  assert len(encodedSingularsTotal) == len(encodedPluralsTotal)
#  redprod1(formations[group])
#  hasVisitedS = (hasVisitedS or (group == "s"))
#  if hasVisitedS:
  #   print(group)
   #  print(dict(encodedSingularsTotal)["band"])
    # print(encodeListOfWords(["."+"band"])[0])
#     redprod1(formations["s"])


#redprod1(formations["s"])
#redprod1(formations["r"])
#redprod1(formations["e"])
#redprod1(formations["n"])
#redprod1(formations["same"])



encodedPluralsTotal = dict(encodedPluralsTotal)
encodedSingularsTotal = dict(encodedSingularsTotal)



#redprod2(formations["s"])
#redprod2(formations["r"])
#redprod2(formations["e"])
#redprod2(formations["n"])
#redprod2(formations["same"])



forNSE = list(plusL([formations["n"], formations["s"], formations["e"]]))

lengthsS = [0 for _ in range(55)]
lengthsP = [0 for _ in range(55)]

for sing, plur in forNSE:
   lengthsS[len(sing)] += 1
   lengthsP[len(plur)] += 1
   

lengths = [min(x,y) for x,y in zip(lengthsS, lengthsP)]

sumLengthsS = sum(lengthsS)
lengthsS = [float(x)/sumLengthsS for x in lengthsS]

sumLengthsP = sum(lengthsP)
lengthsP = [float(x)/sumLengthsP for x in lengthsP]

sumLengths = sum(lengths)
lengths = [float(x)/sumLengths for x in lengths]

ratioP = max([x/y if y > 0 else 0.0 for (x,y) in zip(lengths, lengthsP)])
ratioS = max([x/y if y > 0 else 0.0 for (x,y) in zip(lengths, lengthsS)])

import random

def getResult(stimuli, logisticRegr):
     encodedPluralsR = [encodedPluralsTotal[y] for x, y in stimuli]
     encodedSingularsR = [encodedSingularsTotal[x] for x, y in stimuli]
    
     predictors =  encodedSingularsR + encodedPluralsR
     dependent = [0 for _ in encodedSingularsR] + [1 for _ in encodedPluralsR]
     
     score = logisticRegr.score(predictors, dependent)

     return score



def selectTrainingSet(formations):
     global N
     singulars = {}
     plurals = {}
     for typ in ["n", "s", "e"]:
        singulars[typ] = []
        plurals[typ] = []
     
#        formations[typ] = sorted(list(formations[typ]))
        for _ in range(N):
           while True:
              index, sampledS = random.choice(list(zip(range(len(formations[typ])), formations[typ])))
              sampledS = sampledS[0]
              ratio = lengths[len(sampledS)] / (ratioS * lengthsS[len(sampledS)])
              assert 0<= ratio
              assert ratio <= 1
              if random.random() < ratio:
                  del formations[typ][index]
                  singulars[typ].append(sampledS)
                  break
              
           while True:
              index, sampledP = random.choice(list(zip(range(len(formations[typ])), formations[typ])))
              sampledP = sampledP[1]
              ratio = lengths[len(sampledP)] / (ratioP * lengthsP[len(sampledP)])
              assert 0<= ratio
              assert ratio <= 1
              if random.random() < ratio:
                 del formations[typ][index]
                 plurals[typ].append(sampledP)
                 break
     return singulars, plurals

# from each type, sample N singulars and N plurals
N = 15
evaluationPoints = []

formationsBackup = formations


from sklearn.linear_model import LogisticRegression

random.seed(1)
for _ in range(20):
     formations = {x : sorted(list(y)[:]) for x, y in formationsBackup.items()}
     singulars, plurals = selectTrainingSet(formations)
     plurals = plurals["n"] + plurals["s"] + plurals["e"]
     singulars = singulars["n"] + singulars["s"] + singulars["e"]
     assert len(plurals) == len(singulars)
     
     
     print(singulars)
     print(plurals)
     print(len(plurals)) 
     print(sum([len(x) for x in plurals])/float(len(plurals)))
     print(sum([len(x) for x in singulars])/float(len(singulars)))
     
     encodedPlurals =   [encodedPluralsTotal[x] for x in plurals] #encodeListOfWords(["."+y for y in plurals])
     encodedSingulars = [encodedSingularsTotal[x] for x in singulars] #encodeListOfWords(["."+x for x in singulars])
    
     x_train = encodedSingulars + encodedPlurals 
     y_train = [0 for _ in encodedSingulars] + [1 for _ in encodedPlurals] 
     
     print("regression")
     
     logisticRegr = LogisticRegression()
     
     logisticRegr.fit(x_train, y_train)

     for formation in ["n","s","e","r","same"]:
       score = getResult(formations[formation], logisticRegr)
       print([f"{formation} plurals",score])
       evaluationPoints.append((formation, score))
  
     
    

print("----------------")

import math

firstEntries = list(set([x[0] for x in evaluationPoints]))
print("\t".join(["Type", "Acc", "SD", "Lower", "Upper"]))
for entry in ["n","s","e","r","same"]:
   values = [x[1] for x in evaluationPoints if x[0] == entry]
   accuracy = sum(values)/len(values)
   sd = math.sqrt(sum([x**2 for x in values])/len(values) - accuracy**2)
   values = sorted(values)
   lower = values[int(0.05*len(values))]
   upper = values[int(0.95*len(values))]
   print("\t".join([str(x) for x in [entry, round(100*accuracy,1), round(100*sd,1), round(100*lower,1), round(100*upper,1)]]))


quit()



