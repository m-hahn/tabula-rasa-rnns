from paths import WIKIPEDIA_HOME
from paths import MODELS_HOME


#python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj_WORDS.py --language italian --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 1.2 --lr_decay 0.98 --load-from wiki-italian-nospaces-bptt-words-316412710 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128, 128, 256]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([100, 200, 300]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.8, 0.9, 1.0,1.0,  1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))


import math

args=parser.parse_args()
print(args)


assert "word" in args.load_from, args.load_from

print(args)



import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt"}[args.language]

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

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

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

print(torch.nn.functional.cosine_similarity(out1, out2, dim=0))
#print(torch.nn.functional.cosine_similarity(hidden1, hidden2, dim=0))
#print(torch.nn.functional.cosine_similarity(cell1, cell2, dim=0))

#print("willmach")
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ichmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".dumach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ermach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".siemach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esmach"))))
#
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ichmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".dumach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ermach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".siemach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esdenk"))))
#
def doChoiceList(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x.split(" ")) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    return np.argmin(losses)
def doChoiceListLosses(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x.split(" ")) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    return losses



def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x.split(" ")), encodeWord(y.split(" ")))
    print(losses)
    return 0 if losses[0] < losses[1] else 1




with open("stimuli/candidate_adv_aoadj_testset.txt", "r") as inFile:
    dataset = [tuple(x.split("\t")) for x in inFile.read().strip().split("\n")]

choiceMasc = [0,0]
choiceFem = [0,0]

rejectedDueToOOV = 0
for adverb, adjective in dataset:
    adjectiveA = adjective[:-1]+"a"
    if adverb not in stoi or adjective not in stoi or adjectiveA not in stoi:
        rejectedDueToOOV += 1
        continue
    masculine = [f". il {adverb} {adjective} .", f". il {adverb} {adjectiveA} ."]
    choiceMasc[doChoiceList(masculine)] += 1
    print(choiceMasc[0] / sum(choiceMasc))
    feminine = [f". la {adverb} {adjectiveA} .", f". la {adverb} {adjective} ."]
    choiceFem[doChoiceList(feminine)] += 1
    print(choiceFem[0] / sum(choiceFem))

with open("stimuli/candidate_adv_aeadj_testset.txt", "r") as inFile:
   dataset = [tuple(x.split("\t")) for x in inFile.read().strip().split("\n")]

choiceSg = [0,0]
choicePl = [0,0]

rejectedDueToOOV = 0

for adverb, adjective in dataset:
    adjectiveE = adjective[:-1]+"e"
    if adverb not in stoi or adverb not in stoi or adjectiveE not in stoi:
      rejectedDueToOOV += 1
      continue
    singular = [f". la {adverb} {adjective} .", f". la {adverb} {adjectiveE} ."]
    choiceSg[doChoiceList(singular)] += 1
    print(choiceSg[0] / sum(choiceSg))
    plural = [f". le {adverb} {adjectiveE} .", f". le {adverb} {adjective} ."]
    choicePl[doChoiceList(plural)] += 1
    print(choicePl[0] / sum(choicePl))



with open("stimuli/candidate_eadj_aonoun_testset.txt", "r") as inFile:
   dataset = [tuple(x.split(" ")) for x in inFile.read().strip().split("\n")]

choice2Masc = [0,0]
choice2Fem = [0,0]

rejectedDueToOOV = 0


for adjective, noun in dataset:
    nounA = noun[:-1]+"a"
    if adjective not in stoi or noun not in stoi or nounA not in stoi:
       rejectedDueToOOV += 1
       continue
    masc = [f". il {adjective} {noun} .", f". la {adjective} {noun} ."]
    choice2Masc[doChoiceList(masc)] += 1
    print(choice2Masc[0] / sum(choice2Masc))
    fem = [f". il {adjective} {nounA} .", f". la {adjective} {nounA} ."]
    choice2Fem[doChoiceList(fem)] += 1
    print(choice2Fem[1] / sum(choice2Fem))

print("OOV Stimuli", rejectedDueToOOV/len(dataset))
quit()



print("stimuli/candidate_adv_aoadj_testset.txt")
print(choiceMasc[0] / sum(choiceMasc))
print(choiceFem[0] / sum(choiceFem))
print("stimuli/candidate_adv_aeadj_testset.txt")
print(choiceSg[0] / sum(choiceSg))
print(choicePl[0] / sum(choicePl))
print("stimuli/candidate_eadj_aonoun_testset.txt")
print(choice2Masc[0] / sum(choice2Masc))
print(choice2Fem[1] / sum(choice2Fem))




