from paths import MSR_COMP_HOME
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
parser.add_argument("--train_size", type=int, default=25)



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

with open(MSR_COMP_HOME+"//test_answer.csv", "r") as inFile:
   answers = [x.split(",") for x in inFile.read().strip().split("\n")[1:]]

correct = 0.0
with open(MSR_COMP_HOME+"//testing_data.csv", "r") as inFile:
   completion = inFile.read().strip().split("\n")[1:]
for i in range(len(completion)):
  x = completion[i].split(",")
  number = x[0]
  assert number == answers[i][0]
  options = x[-5:]
  assert len(options) == 5
  sentence = (",".join(x[1:-5])).lower()
  if sentence[0] == '"' and sentence[-1] == '"':
      sentence = sentence[1:-1]
  

 # print(options)
#  print(sentence)
  sentences = ["."+sentence.replace("_____", x).replace(" ","") for x in options]
#  print(sentences[0])
  chosen = doChoiceList(sentences)
  chosenChar = "abcde"[chosen]
  correct += (1 if chosenChar == answers[i][1] else 0)
  print(correct/(i+1))

quit()







with open("/private/home/mhahn/data/similarity/fullVocab.txt", "r") as inFile:
   testVocabulary = set(inFile.read().strip().split("\n"))

with open("/private/home/mhahn/data/similarity/MEN/MEN_dataset_natural_form_full", "r") as inFile:
   simlist = [x.split(" ") for x in inFile.read().strip().split("\n")[1:]]
for sim in simlist:
  testVocabulary.add(sim[0])
  testVocabulary.add(sim[1])


testVocabulary = sorted(list(testVocabulary), key=lambda x:len(x))


i = 0
length = 0
wordVectors = {}

while i < len(testVocabulary):
   toEncode = [testVocabulary[i]]
   assert len(testVocabulary[i]) > length, (i, len(testVocabulary[i]), length)
   length = len(testVocabulary[i])
   while i+1 < len(testVocabulary):
      i += 1
      if len(testVocabulary[i]) == length:
         toEncode.append(testVocabulary[i])
      else:
         assert len(testVocabulary[i]) > length
         break
   else:
      assert i+1 == len(testVocabulary)
      i += 1
   assert i == len(testVocabulary) or len(testVocabulary[i]) > length

   vectors = encodeListOfWords(toEncode)
   if True:
      print(i)
   for word, vector in zip(toEncode, vectors):
       if False:
          print(" ".join(list(map(str,[word] + list(vector)))))
       wordVectors[word] = vector

import torch.nn.functional
import numpy as np
from scipy import spatial


def computeCosine(word1, word2):
   vector0 = encodeListOfWords([word1])[0]
   vector1 = encodeListOfWords([word2])[0]
   print(word1, word2, 1-spatial.distance.cosine(vector0, vector1))

computeCosine("wine", "wine7")

computeCosine("wine", "wines")
computeCosine(".wine", "wine")
computeCosine("wine", "thewine")
computeCosine(".wine", "the.wine")
computeCosine("wine", "iwne")
computeCosine("wine", "inwe")
computeCosine("wine", "niwe")
computeCosine("wine", "beer")
computeCosine("computer", "laptop")
computeCosine(",good,", ",wonderful,")
computeCosine(",good,", ",computer,")

#quit()

cosines = []
sims = []
with open("/private/home/mhahn/data/similarity/wordsim353/combined.csv", "r") as inFile:
   simlist = [x.split(",") for x in inFile.read().strip().split("\n")[1:]]
for sim in simlist:
   vec1 = wordVectors[sim[0].lower()]
   vec2 = wordVectors[sim[1].lower()]
   cosine = spatial.distance.cosine(vec1, vec2)
   cosines.append(1-cosine)
   sims.append(float(sim[2]))
 #  print(cosine, sims[-1])

import scipy.stats
print(scipy.stats.spearmanr(cosines, sims))



cosines = []
sims = []
with open("/private/home/mhahn/data/similarity/MTURK-771.csv", "r") as inFile:
   simlist = [x.split(",") for x in inFile.read().strip().split("\n")[1:]]
for sim in simlist:
   vec1 = wordVectors[sim[0].lower()]
   vec2 = wordVectors[sim[1].lower()]
   cosine = spatial.distance.cosine(vec1, vec2)
   cosines.append(1-cosine)
   sims.append(float(sim[2]))
#   print(cosine, sims[-1])

import scipy.stats
print(scipy.stats.spearmanr(cosines, sims))



cosines = []
sims = []
with open("/private/home/mhahn/data/similarity/MEN/MEN_dataset_lemma_form.dev", "r") as inFile:
   simlist = [x.split(" ") for x in inFile.read().strip().split("\n")[1:]]
for sim in simlist:
   vec1 = wordVectors[sim[0][:-2].lower()]
   vec2 = wordVectors[sim[1][:-2].lower()]
   cosine = spatial.distance.cosine(vec1, vec2)
   cosines.append(1-cosine)
   sims.append(float(sim[2]))
#   print(cosine, sims[-1])

import scipy.stats
print(scipy.stats.spearmanr(cosines, sims))



projection = torch.nn.Linear(args.hidden_dim, 100)
#projection = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1).cuda()

final = torch.nn.Linear(1, 1)

embeddingsCNLM = torch.nn.Embedding(num_embeddings=len(testVocabulary), embedding_dim=args.hidden_dim)
for i in range(len(testVocabulary)):
  embeddingsCNLM.weight.data[i] = torch.FloatTensor(wordVectors[testVocabulary[i]])


modules = [projection, final]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=10.0, momentum=0.0) # 0.02, 0.9
for i in range(2002):
   product = torch.mm(projection.weight , torch.transpose(projection.weight, 0, 1))
   loss = torch.nn.MSELoss()(product, torch.eye(100))

#   product = torch.mm( torch.transpose(projection.weight, 0, 1), projection.weight)
#   loss += torch.nn.MSELoss()(product, torch.eye(1024))

   if i % 1000 == 0:
     print(loss)
   optim.zero_grad()
   loss.backward()
   optim.step()


optim = torch.optim.SGD(parameters(), lr=0.1, momentum=0.9) # 0.02, 0.9

#quit()


simlist = [(x[0][:-2], x[1][:-2], float(x[2])/50) for x in simlist]
train = simlist[100:]
dev = simlist[:100]

itosSim = dict(zip(testVocabulary, range(len(testVocabulary))))

def forwardProject(pair, train):
  assert itosSim[pair[0]] < len(testVocabulary)
#  print(embeddingsCNLM)
 # print(itosSim[pair[0]])
  emb1 = embeddingsCNLM(torch.LongTensor([itosSim[pair[0]]]).view(1,1,-1)).view(-1)
  emb2 = embeddingsCNLM(torch.LongTensor([itosSim[pair[1]]]).view(1,1,-1)).view(-1)
  if train:
    mask = Variable(torch.bernoulli(emb1.data.new(emb1.data.size()).fill_(0.4)))
    emb1 = emb1 * mask / 0.4
    emb2 = emb2 * mask / 0.4

  proj1 = projection(emb1)
  proj2 = projection(emb2)
  norm1 = torch.sqrt(torch.dot(proj1, proj1))
  norm2 = torch.sqrt(torch.dot(proj2, proj2))
#  print("Norm", norm1 * norm2)
 # print("Dot", torch.dot(proj1, proj2))
  similarity = final(torch.div(torch.dot(proj1, proj2) , (norm1 * norm2 + 1e-8)).view(1))

  loss = torch.nn.MSELoss()(pair[2], similarity)



  product = torch.mm(projection.weight , torch.transpose(projection.weight, 0, 1))
  lossOrthog =0.1 *  torch.nn.MSELoss()(product, torch.eye(100))
#  print(lossOrthog)
  loss += lossOrthog


  return loss
  #print(similarity.data.numpy(), pair[2], loss.data.numpy())

def backwardProject(loss):
    optim.zero_grad()
    loss.backward()
    optim.step()

import random
for epoch in range(100):
   loss = 0
   for pair in dev:
      loss += forwardProject(pair, train=False).data.cpu().numpy()
   loss /= len(dev)
   print(epoch, loss)
   print(final.weight)
   print(final.bias)

   cosines = []
   sims = []
   for sim in simlist:
      vec1 = projection(torch.FloatTensor(wordVectors[sim[0]])).data.numpy()
      vec2 = projection(torch.FloatTensor(wordVectors[sim[1]])).data.numpy()
      cosine = spatial.distance.cosine(vec1, vec2)
      cosines.append(1-cosine)
      sims.append(float(sim[2]))
   #   print(cosine, sims[-1])
   
   import scipy.stats
   print(scipy.stats.spearmanr(cosines, sims))
   
   
   cosines = []
   sims = []
   with open("/private/home/mhahn/data/similarity/MTURK-771.csv", "r") as inFile:
      simlist771 = [x.split(",") for x in inFile.read().strip().split("\n")[1:]]
   for sim in simlist771:
      vec1 = projection(torch.FloatTensor(wordVectors[sim[0].lower()])).data.numpy()
      vec2 = projection(torch.FloatTensor(wordVectors[sim[1].lower()])).data.numpy()
      cosine = spatial.distance.cosine(vec1, vec2)
      cosines.append(1-cosine)
      sims.append(float(sim[2]))
   #   print(cosine, sims[-1])
   
   import scipy.stats
   print(scipy.stats.spearmanr(cosines, sims))
   




   random.shuffle(train)
   counter = 0
   for pair in train:
      counter += 1
      if counter % 100 == 0:
        print("in epoch", counter/len(train))
      loss = forwardProject(pair, train=True)
      backwardProject(loss)


