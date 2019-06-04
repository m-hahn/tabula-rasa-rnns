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

parser.add_argument("--batchSize", type=int, default=random.choice([128, 128, 256]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([50, 100, 200, 200]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([256, 512, 1024, 2048]))
parser.add_argument("--layer_num", type=int, default=random.choice([1,2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.01, 0.01, 0.1, 0.2]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([10, 20, 30, 50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))
parser.add_argument("--nonlinearity", type=str, default=random.choice(["tanh", "relu"]))
parser.add_argument("--train_size", type=int, default=25)




import math

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


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

def encodeWord(word):
      numeric = [[]]
      for char in word:
           numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric



rnn_drop.train(False)
#rnn_forward_drop.train(False)
#rnn_backward_drop.train(False)

#baseline_rnn_encoder_drop.train(False)

lossModule = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)


def choice(numeric1, numeric2):
     assert len(numeric1) == 1
     assert len(numeric2) == 1
     numeric = [numeric1[0], numeric2[0]]
     maxLength = max([len(x) for x in numeric])
     for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
              numeric[i].append(0)
     input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
     
     target = input_tensor_forward[1:]
     input_cut = input_tensor_forward[:-1]
     embedded_forward = char_embeddings(input_cut)
     out_forward, hidden_forward = rnn_drop(embedded_forward, None)

     prediction = logsoftmax(output(out_forward)) #.data.cpu().view(-1, 3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     losses = lossModule(prediction.view(-1, len(itos)+3), target.view(-1)).view(maxLength, 2)
     losses = losses.sum(0).data.cpu().numpy()
     return losses


def encodeListOfWords(words):
    numeric = [encodeWord(word)[0] for word in words]
    maxLength = max([len(x) for x in numeric])
    for i in range(len(numeric)):
       numeric[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
    
    input_cut = input_tensor_forward #[:-1]
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)
    hidden = hidden_forward.data.cpu().numpy()
    return [hidden[0][i] for i in range(len(words))]




def choiceList(numeric):
     for x in numeric:
       assert len(x) == 1
#     assert len(numeric1) == 1
 #    assert len(numeric2) == 1
     numeric = [x[0] for x in numeric] #, numeric2[0]]
     maxLength = max([len(x) for x in numeric])
     for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
              numeric[i].append(0)
     input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
     
     target = input_tensor_forward[1:]
     input_cut = input_tensor_forward[:-1]
     embedded_forward = char_embeddings(input_cut)
     out_forward, hidden_forward = rnn_drop(embedded_forward, None)

     prediction = logsoftmax(output(out_forward)) #.data.cpu().view(-1, 3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     losses = lossModule(prediction.view(-1, len(itos)+3), target.view(-1)).view(maxLength, len(numeric))
     losses = losses.sum(0).data.cpu().numpy()
     return losses



def encodeSequenceBatchForward(numeric):
      input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)

#      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_forward = char_embeddings(input_tensor_forward)
      out_forward, hidden_forward = rnn_drop(embedded_forward, None)
#      out_forward = out_forward.view(args.sequence_length+1, len(numeric), -1)
 #     logits_forward = output(out_forward) 
  #    log_probs_forward = logsoftmax(logits_forward)
      return (out_forward[-1], hidden_forward)



def encodeSequenceBatchBackward(numeric):
#      print([itos[x-3] for x in numeric[0]])
#      print([[0]+(x[::-1]) for x in numeric])
      input_tensor_backward = Variable(torch.LongTensor([[0]+(x[::-1]) for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
#      target_tensor_backward = Variable(torch.LongTensor([x[::-1] for x in numeric]).transpose(0,1)[:-2].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_backward = char_embeddings(input_tensor_backward)
      out_backward, hidden_backward = rnn_backward_drop(embedded_backward, None)
#      out_backward = out_backward.view(args.sequence_length+1, len(numeric), -1)
#      logits_backward = output(out_backward) 
#      log_probs_backward = logsoftmax(logits_backward)

      return (out_backward[-1], hidden_backward)


import numpy as np

def predictNext(encoded, preventBoundary=True):
     out, hidden = encoded
     prediction = logsoftmax(output(out.unsqueeze(0))).data.cpu().view(3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     predicted = np.argmax(prediction[:-1] if preventBoundary else prediction)
     return itos[predicted-3] #, prediction

def keepGenerating(encoded, length=100, backwards=False):
    out, hidden = encoded
    output_string = ""
   
#    rnn_forward_drop.train(True)

    for _ in range(length):
      prediction = logsoftmax(2*output(out.unsqueeze(0))).data.cpu().view(3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
#      predicted = np.argmax(prediction).items()
      predicted = np.random.choice(3+len(itos), p=np.exp(prediction))

      output_string += itos[predicted-3]

      input_tensor_forward = Variable(torch.LongTensor([[predicted]]).transpose(0,1).cuda(), requires_grad=False)

      embedded_forward = char_embeddings(input_tensor_forward)
      
      out, hidden = (rnn_drop if not backwards else rnn_backward_drop)(embedded_forward, hidden)
      out = out[-1]

 #   rnn_forward_drop.train(False)


    return output_string if not backwards else output_string[::-1]


vocabPath = {"german" : WIKIPEDIA_HOME+"german-wiki-word-vocab-POS.txt", "italian" : WIKIPEDIA_HOME+"itwiki/italian-wiki-word-vocab-POS.txt"}[args.language]


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

nouns = [x for x in nouns if x not in verbsTotal]
verbs = [x for x in verbs if x not in nounsTotal]
print(len(nouns))
print(len(verbs))


def criterion(word):
    if args.language == "german":
       return word.endswith("en")
    elif args.language == "italian":
       return word.endswith("re")

nounsInN = [x for x in nouns if criterion(x)]
verbsInN = [x for x in verbs if criterion(x)]
print(len(nounsInN))
print(len(verbsInN))

print(nounsInN[:100])
print(verbsInN[:100])

sampleSize = 500

accuracies = []
for _ in range(100):
  random.shuffle(nounsInN)
  random.shuffle(verbsInN)


  nounsInNSelected = nounsInN[:sampleSize]# len(verbsInN)]
  verbsInNSelected = verbsInN[:sampleSize] #len(verbsInN)]
  
  encodedNounsInN = encodeListOfWords([x for x in nounsInNSelected])
  encodedVerbsInN = encodeListOfWords([x for x in verbsInNSelected])
  
  predictors = encodedNounsInN + encodedVerbsInN
  
  dependent = [0 for _ in encodedNounsInN] + [1 for _ in encodedVerbsInN]
  
  from sklearn.model_selection import train_test_split
  x_train_nouns, x_test_nouns, y_train_nouns, y_test_nouns = train_test_split(encodedNounsInN, [0 for _ in encodedNounsInN], test_size=1-args.train_size/500, shuffle=True)
  x_train_verbs, x_test_verbs, y_train_verbs, y_test_verbs = train_test_split(encodedVerbsInN, [1 for _ in encodedVerbsInN], test_size=1-args.train_size/500, shuffle=True)

  x_train = x_train_nouns + x_train_verbs
  y_train = y_train_nouns + y_train_verbs
  x_test = x_test_nouns + x_test_verbs
  y_test = y_test_nouns + y_test_verbs



  from sklearn.linear_model import LogisticRegression
  
  print("regression")
  
  logisticRegr = LogisticRegression()
  
  logisticRegr.fit(x_train, y_train)
  
  score = logisticRegr.score(x_test, y_test)
  print(score)
  accuracies.append(score)

print("--")

print(sum(accuracies)/100)

meanAccuracy = sum(accuracies)/100
meanSquaredAccuracy = sum([x**2 for x in accuracies])/100
import math
standardDeviation = math.sqrt(meanSquaredAccuracy - meanAccuracy**2)

print(standardDeviation)


accuracies = sorted(accuracies)

ci_lower = accuracies[int(0.05 * 100)]
ci_upper = accuracies[int(0.95 * 100)]

print((ci_lower, ci_upper))

with open(f"/checkpoint/mhahn/pos/{__file__}_"+args.language+"_"+str(args.train_size)+"_"+args.load_from, "w") as outFile:
   print(args.language, file=outFile)
   print(args.train_size, file=outFile)
   print(args.load_from, file=outFile)
   print(sum(accuracies)/100, file=outFile)  
   print(standardDeviation)
   print(ci_lower)
   print(ci_upper)


