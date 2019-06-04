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



import math

args=parser.parse_args()
print(args)





import corpusIteratorWiki



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


word_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt"}[args.language]

with open(word_vocab_path, "r") as inFile:
     itos_words = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi_words = dict([(itos_words[i],i) for i in range(len(itos_words))])



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

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

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


out1, hidden1 = encodeSequenceBatchForward(encodeWord("katze"))
out2, hidden2 = encodeSequenceBatchForward(encodeWord("katzem"))
#print(torch.dot(out1[-1], out2[-1]))
#print(torch.dot(hidden1[0], hidden2[0]))
#print(torch.dot(hidden1[1], hidden2[1]))

def doChoiceList(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    return np.argmin(losses)
def doChoiceListLosses(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    return losses



def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x), encodeWord(y))
    print(losses)
    return 0 if losses[0] < losses[1] else 1

from corpusIterator import CorpusIterator

adjectives = []
wentThroughAdjectives = False
with open(WIKIPEDIA_HOME+"german-wiki-word-vocab-lemmas-POS-uniq.txt", "r") as inFile:
    adjectives = []
    for line in inFile:
      line = line.strip().split(" ")
      if len(line) != 3:
        continue
      if line[1] != "ADJA":
          if wentThroughAdjectives:
             continue
      else:
        wentThroughAdjectives = True
      if line[2] == "<unknown>":
         continue
      if len(line[2]) == 1:
        continue
      if "." in line[2]:
        continue
      if int(line[0]) > 100 and not line[2].endswith("r"):
         if line[2]+"e" in stoi_words:
            adjectives.append(line[2])


def genderTest(mode):
   training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True)
   genders = dict([("Gender="+x, set()) for x in ["Masc", "Fem", "Neut"]])
   for sentence in training.iterator():
       for line in sentence:
        if line["posUni"] == "NOUN" and "|" not in line["lemma"]:
        
           morph = line["morph"]
           if "Number=Sing" in morph and "Case=Nom" in morph:
            gender = [x for x in morph if x.startswith("Gender=")]
            if len(gender) > 0:
              genders[gender[0]].add(line["lemma"].lower())
              
   #print(genders)
   counter = 0

   results = [[0,0,0] for _ in range(3)]
   for genderIndex, gender in enumerate(["Gender="+x for x in ["Masc", "Fem", "Neut"]]):
     with open(f"stimuli/german-gender-{gender}-{mode}.txt", "w") as outFile:
       counter = 0
       for noun in genders[gender]:
         if noun not in stoi_words:
            continue
         counter += 1
     #    adverbs = ["sehr"]
      #   adjective = "" #"".join(adverbs)+random.choice(adjectives)+"e"
         if mode == "nothing":
           noun = noun
           nounStimulus = [noun]
         elif mode == "adjective":
            adjective = random.choice(adjectives)+"e"
            nounStimulus = [adjective, noun]
            noun = adjective+noun
         elif mode == "sehr + adjective":
            adjective = random.choice(adjectives)+"e"
            nounStimulus = ["sehr", adjective, noun]
            noun = "sehr"+adjective+noun
         elif mode == "sehr + extrem + adjective":
            adjective = random.choice(adjectives)+"e"
            nounStimulus = ["sehr", "extrem", adjective, noun]
            noun = "sehr"+"extrem"+adjective+noun


 
         stimuli = []
         print(" ".join(["der"] + nounStimulus), file=outFile)
         print(" ".join(["die"] + nounStimulus), file=outFile)
         print(" ".join(["das"] + nounStimulus), file=outFile)
         
 
  #       noun = f"{adjective}{noun}"
         results[genderIndex][doChoiceList([f".der{noun}.", f".die{noun}.", f".das{noun}."], printHere=(random.random() > 0.98))] += 1
  #       results[doChoiceList([".ein"+noun+".", ".eine"+noun+"."])] += 1
         if random.random() > 0.98:
            print([[round(x/(counter if genderIndex == i else 1), 2) for x in results[i]] for i in range(len(results))])
       results[genderIndex] = [x/counter for x in results[genderIndex]]
   return results


#   # test separation of feminine from masc/neuter via indefinite
#   results = [0,0,0] 
#   for noun in genders["Gender=Masc"].union(genders["Gender=Neut"]):
#       counter += 1
##       results[doChoiceList([".der"+noun+".", ".die"+noun+".", ".das"+noun+"."])] += 1
#       results[doChoiceList([".ein"+noun+".", ".eine"+noun+"."])] += 1
#       print([x/counter for x in results])
#   return [x/counter for x in results]
#
confusion1 = genderTest("nothing")
confusion2 = genderTest("adjective")
confusion3 = genderTest("sehr + adjective")
confusion4 = genderTest("sehr + extrem + adjective")

print(confusion1)
print(confusion2)
print(confusion3)
print(confusion4)





import numpy as np
losses  = (doChoiceListLosses([".der", ".die", ".das"]))
losses = np.exp(-losses)
print(losses/np.sum(losses))


