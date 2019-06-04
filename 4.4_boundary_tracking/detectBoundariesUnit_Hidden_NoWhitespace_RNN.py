#python detectBoundariesUnit_Hidden_NoWhitespace_RNN.py --batchSize 256 --char_dropout_prob 0.01 --char_embedding_size 50 --char_noise_prob 0.0 --hidden_dim 2048 --language german --layer_num 2 --learning_rate 0.1 --lr_decay 0.95 --nonlinearity tanh --load-from wiki-german-nospaces-bptt-rnn-237671415 --sequence_length 30 --verbose True --weight_dropout_hidden 0.0 --weight_dropout_in 0.0
#tensor([0.4606, 0.4367, 0.4249, 0.4115, 0.3972]) tensor([ 764,  450, 1742, 1635,  752]) 

#python detectBoundariesUnit_Hidden_NoWhitespace_RNN.py --batchSize 256 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language italian --layer_num 2 --learning_rate 0.004 --lr_decay 0.98 --nonlinearity tanh --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0
#tensor([0.5356, 0.5191, 0.4944, 0.4610, 0.4562]) tensor([1786,  309, 1421, 1050, 1961])

#python detectBoundariesUnit_Hidden_NoWhitespace_RNN.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
# tensor([0.3986, 0.3965, 0.3789, 0.3696, 0.3685]) tensor([ 730,  676,  224, 1960, 2010])


from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)

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
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))
parser.add_argument("--nonlinearity", type=str, default=random.choice(["tanh", "relu"]))




args=parser.parse_args()
print(args)


#assert args.language == "german"


import corpusIteratorWiki



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    assert False
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
#itos.append(" ")
#assert " " in itos
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.RNN(args.char_embedding_size, args.hidden_dim, args.layer_num, args.nonlinearity).cuda()

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

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      boundariesAll = [None for _ in range(args.sequence_length+1)]

      count = 0
      currentWord = ""
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
         if char == " ":
           boundaries[len(numeric)] = currentWord
           boundariesAll[len(numeric)] = currentWord

           currentWord = ""
           continue
         else:
           if boundariesAll[len(numeric)] is None:
               boundariesAll[len(numeric)] = currentWord

         count += 1
         currentWord += char
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric, boundaries, boundariesAll
            numeric = [0]
            boundaries = [None for _ in range(args.sequence_length+1)]
            boundariesAll = [None for _ in range(args.sequence_length+1)]





#def prepareDataset(data, train=True):
#      numeric = [0]
#      count = 0
#      for char in data:
#         if char == " ":
#           continue
#         count += 1
##         if count % 100000 == 0:
##             print(count/len(data))
#         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
#         if len(numeric) > args.sequence_length:
#            yield numeric
#            numeric = [0]
#

# from each bath element, get one positive example OR one negative example

wordsSoFar = set()
hidden_states = []
labels = []
relevantWords = []
relevantNextWords = []
labels_sum = 0

def forward(numeric, train=True, printHere=False):
      global labels_sum
      numeric, boundaries, boundariesAll = zip(*numeric)


      selected = []
      for i in range(len(boundaries)): # for each batch sample
         target = (labels_sum + 10 < len(labels)*0.7) or (random.random() < 0.5) # decide whether to get positive or negative sample
         true = sum([((x == None) if target == False else (x is not None and y not in wordsSoFar)) for x, y in list(zip(boundaries[i], boundariesAll[i]))[int(args.sequence_length/2):-10]]) # condidates
 #        print(target, true)
         if true == 0:
            continue
         soFar = 0
         for j in range(int(len(boundaries[i])/2), len(boundaries[i])-10):
           if (lambda x, y:((x is None if target == False else (x is not None and y not in wordsSoFar))))(boundaries[i][j], boundariesAll[i][j]):
              if random.random() < 1/(true-soFar):
                  selected.append((len(selected),i,j,target))
                  assert (boundaries[i][j] is not None) == target, (boundaries[i][j], boundariesAll[i][j], target)
                  break
              soFar += 1
         assert soFar < true

      if len(selected) == 0:
         return

      numeric_selected = []
      for _,i,j,_ in selected:
        numeric_selected.append(numeric[i][j-10:j+1]) # do not include the actual boundary
#      print(numeric_selected)
      input_tensor = Variable(torch.LongTensor(numeric_selected).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric_selected).transpose(0,1)[1:].cuda(), requires_grad=False)

      embedded = char_embeddings(input_tensor)
      if train:
         #assert False
         embedded = char_dropout(embedded)

#      print(embedded.size())
      out, hidden = rnn_drop(embedded, None)

      for i,i2,j,target in selected:
                  assert i < len(numeric_selected)
      #            print(hidden[0].size())
#                  quit()
                  hidden_states.append(hidden[1][i,:].detach().data.cpu().numpy())
                  labels.append(1 if target else 0)
                  relevantWords.append(boundariesAll[i2][j])
                  relevantNextWords.append(([boundaries[i2][k] for k in range(j+1, len(boundaries[i2])) if boundaries[i2][k] is not None]+["END_OF_SEQUENCE"])[0])
#                  print(target, relevantWords[-1], relevantNextWords[-1])
                  assert boundariesAll[i2][j] is not None

                  labels_sum += labels[-1]
##      print(hidden_states)
##      print(labels)
#
#      logits = output(out) 
#      log_probs = logsoftmax(logits)
#   #   print(logits)
#  #    print(log_probs)
# #     print(target_tensor)
#
#      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
#
#      if printHere:
#         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(args.sequence_length, len(numeric_selected))
#         losses = lossTensor.data.cpu().numpy()
##         boundaries_index = [0 for _ in numeric]
#         for i in range((args.sequence_length-1)-1):
# #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
#  #             boundary = True
#   #            boundaries_index[0] += 1
#    #        else:
#     #          boundary = False
#            print((losses[i][0], itos[numeric[0][i+1]-3], "read:", itos[numeric[0][i]-3], boundariesAll[0][i], boundariesAll[0][i+1] if i < args.sequence_length-2 else "EOS"))
#         print((labels_sum, len(labels)))
     # return loss, len(numeric) * args.sequence_length



import time

devLosses = []
#for epoch in range(10000):
if True:
   training_data = corpusIteratorWiki.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_drop.train(False)
   startTime = time.time()
   trainChars = 0
   counter = 0
   while True:
      counter += 1
      try:
         numeric = [next(training_chars) for _ in range(args.batchSize)]
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      forward(numeric, printHere=printHere, train=True)
      #backward(loss, printHere)
      if printHere:
          print((counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

      if len(labels) > 10000:
         break

predictors = torch.Tensor(hidden_states)
dependent = torch.Tensor(labels).unsqueeze(1)

print(predictors.size())
print(dependent.size())

correlations = ((predictors - predictors.mean(dim=0))*(dependent - 0.5)).mean(dim=0)
sd1 = torch.sqrt(torch.pow(predictors,2).mean(dim=0) - torch.pow(predictors.mean(dim=0),2))
sd2 = torch.sqrt(torch.pow(dependent,2).mean(dim=0) - torch.pow(dependent.mean(dim=0),2))
print(correlations.size())
correlations = correlations/(sd1*sd2)
print(list(correlations))
print(torch.max(correlations))
print(torch.min(correlations))

x,y = correlations.max(0)
print(x,y)
x,y = correlations.min(0)
print(x,y)

x, y = correlations.abs().topk(5)
print(x,y)
for i in y:
   print(y, correlations[i])

