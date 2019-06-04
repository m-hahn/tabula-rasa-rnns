from paths import MSR_COMP_HOME
from paths import LOG_HOME
from paths import MODELS_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([1024]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.1, 0.15, 0.15, 0.15, 0.2]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.01, 0.01, 0.01]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0,  1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5, 1.6]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--clip_grad_limit", type=float, default=random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 10.0, 20.0]))



import math

args=parser.parse_args()

if "MYID" in args.save_to:
   args.save_to = args.save_to.replace("MYID", str(args.myID))

assert "word" in args.save_to, args.save_to

print(args)



import corpusIteratorHolmesWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path=MSR_COMP_HOME+"//holmes-word-vocab.txt" 
with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
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


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings} #, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

parameters_cached = [x for x in parameters()]



from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      for chunk in data:
       #print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numerified.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
       #  if len(numeric) > args.sequence_length:
        #    yield numeric
         #   numeric = [0]
  #     print(len(numerified))
 #      print(args.batchSize)
#       print(args.sequence_length)

       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length
#         elif len(numerified) > args.batchSize:
#            print("Taking small sequence")
#            sequenceLengthHere = int(len(numerified) / args.batchSize)
#            assert sequenceLengthHere < args.sequence_length
#            assert  sequenceLengthHere > 0

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerified = numerified[cutoff:]
        
         #print(len(numerifiedCurrent))
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         #print(numerifiedCurrent.size())
         #quit()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
  #           print(numerifiedCurrent[i].size())
             yield numerifiedCurrent[i]
         hidden = None
       else:
         print("Skipping")

def prepareDatasetChunksPrevious(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
         if char == " ":
           continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]






def prepareDataset(data, train=True):
      numeric = [0]
      count = 0
      for char in data:
         if char == " ":
           continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]

hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

def forward(numeric, train=True, printHere=False):
      global hidden
      global beginning
      if hidden is None or (train and random.random() > 0.9):
          hidden = None
          beginning = zeroBeginning
      elif hidden is not None:
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric = torch.cat([beginning, numeric], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)
      

    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      
      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
#         boundaries_index = [0 for _ in numeric]
         print(("NONE", itos[numericCPU[0][0]-3]))
         for i in range((args.sequence_length)):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            print((losses[i][0], itos[numericCPU[i+1][0]-3]))
      return loss, target_tensor.view(-1).size()[0]

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, args.clip_grad_limit) #, norm_type="inf")
      optim.step()




import time

devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = corpusIteratorHolmesWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   while True:
      counter += 1
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts 
      if printHere:
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(args)
      if counter % 20000 == 0 and epoch <= 1:
        if args.save_to is not None:
           torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")


   rnn_drop.train(False)


   dev_data = corpusIteratorHolmesWords.dev(args.language)
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=False)


     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
   with open(LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
      print(" ".join([str(x) for x in devLosses]), file=outFile)
      print(" ".join(sys.argv), file=outFile)
      print(str(args), file=outFile)
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if devLosses[-1] > 10.0:
      print("Bad loss, aborting")
      quit()
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")

   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9


