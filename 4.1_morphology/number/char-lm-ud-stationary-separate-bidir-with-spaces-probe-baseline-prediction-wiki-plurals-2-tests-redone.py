# Like in the paper (1st round, but more efficient implementation).

# python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-redone.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

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
    
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)
    hidden = hidden_forward[0].data.cpu().numpy()
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





plurals = set()


formations = {"e" : set(), "n" : set(), "s" : set(), "same" : set(), "r" : set()}

for group in formations:
  with open(f"stimuli/german-plurals-{group}.txt", "r") as inFile:
     formations[group] = [tuple(x.split(" ")) for x in inFile.read().strip().split("\n")]
     print(len(formations[group]))




print(formations["n"])
print(formations["same"])

def doChoiceList(xs):
    for x in xs:
       print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    print(losses)
    return np.argmin(losses)


def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x), encodeWord(y))
    print(losses)
    return 0 if losses[0] < losses[1] else 1


# classify singulars vs plurals
print("trained on n, s, e")

    

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

# from each type, sample N singulars and N plurals
N = 15
evaluationPoints = []


encodedPluralsSame = encodeListOfWords(["."+y for x, y in formations["same"]])
encodedSingularsSame = encodeListOfWords(["."+x for x, y in formations["same"]])

encodedPluralsR = encodeListOfWords(["."+y for x, y in formations["r"]])
encodedSingularsR = encodeListOfWords(["."+x for x, y in formations["r"]])


formationsBackup = formations

random.seed(1)
for _ in range(20):
     formations = {x : set(list(y)[:]) for x, y in formationsBackup.items()}


     singulars = {}
     plurals = {}
     for typ in ["n", "s", "e"]:
        singulars[typ] = []
        plurals[typ] = []
     
        formations[typ] = sorted(list(formations[typ]))
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
     
     stratify_types = ["n" for _ in plurals["n"]] + ["s" for _ in plurals["s"]] + ["e" for _ in plurals["e"]]
     
     plurals = plurals["n"] + plurals["s"] + plurals["e"]
     singulars = singulars["n"] + singulars["s"] + singulars["e"]
     
     assert len(plurals) == len(singulars)
     
     
     print(singulars)
     print(plurals)
     print(len(plurals)) 
     print(sum([len(x) for x in plurals])/float(len(plurals)))
     print(sum([len(x) for x in singulars])/float(len(singulars)))
     
     
     encodedPlurals = encodeListOfWords(["."+y for y in plurals])
     encodedSingulars = encodeListOfWords(["."+x for x in singulars])
     
     #predictors = encodedSingulars + encodedPlurals
     
     #dependent = [0 for _ in encodedSingulars] + [1 for _ in encodedPlurals]
     
     from sklearn.model_selection import train_test_split
     sx_train, sx_test, sy_train, sy_test, st_train, st_test = train_test_split(encodedSingulars, [0 for _ in encodedSingulars], stratify_types, test_size=0.5, shuffle=True, stratify = stratify_types, random_state=1) # random_state=random.randint(0,100), 
     px_train, px_test, py_train, py_test, pt_train, pt_test = train_test_split(encodedPlurals, [1 for _ in encodedPlurals], stratify_types, test_size=0.5,  shuffle=True, stratify = stratify_types, random_state=1) # random_state=random.randint(0,100),
     
     x_train = sx_train + px_train
     x_test = sx_test + px_test
     y_train = sy_train + py_train
     y_test = sy_test + py_test
     t_train = st_train + pt_train
     t_test = st_test + pt_test
     
     
     print(y_train)
     print(y_test)
     
     
     from sklearn.linear_model import LogisticRegression
     
     print("regression")
     
     logisticRegr = LogisticRegression()
     
     logisticRegr.fit(x_train, y_train)
     
     predictions = logisticRegr.predict(x_test)
     
     for typ in ["n", "s", "e"]:
      indicesForType = [i for i in range(len(t_test)) if t_test[i] == typ]
      
      print(len(indicesForType))
      score = logisticRegr.score([x_test[i] for i in indicesForType], [y_test[i] for i in indicesForType])
      print([f"test on {typ}",score])
     
      evaluationPoints.append((typ, score))
     
     #print(formations["r"])
     # test on R plurals
     
     predictors =  encodedSingularsR + encodedPluralsR
     dependent = [0 for _ in encodedSingularsR] + [1 for _ in encodedPluralsR]
     
     score = logisticRegr.score(predictors, dependent)
     print(["r plurals",score])
     
     evaluationPoints.append(("r", score))

#     predictions = logisticRegr.predict(predictors)
 #    print(predictions)     

     
     # test on R plurals
     
     predictors = encodedSingularsSame + encodedPluralsSame 
     dependent = [0 for _ in encodedSingularsSame] + [1 for _ in encodedPluralsSame]
     
     score = logisticRegr.score(predictors, dependent)
     print(["same length plurals", score])
     
     evaluationPoints.append(("same", score))


  #   predictions = logisticRegr.predict(predictors)
   #  print(predictions)     
     
     
     

print("----------------")

import math

firstEntries = list(set([x[0] for x in evaluationPoints]))
for entry in firstEntries:
   values = [x[1] for x in evaluationPoints if x[0] == entry]
   accuracy = sum(values)/len(values)
   sd = math.sqrt(sum([x**2 for x in values])/len(values) - accuracy**2)
   values = sorted(values)
   lower = values[int(0.05*len(values))]
   upper = values[int(0.95*len(values))]
   print(entry, accuracy, sd, lower, upper)


quit()



