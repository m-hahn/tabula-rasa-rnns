from paths import WIKIPEDIA_HOME


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)

import random

import math

args=parser.parse_args()
print(args)


import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt"}[args.language]

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t") for x in inFile.read().strip().split("\n")]
stoi = dict([(itos[i][0],int(itos[i][1])) for i in range(len(itos))])


import random


import numpy as np



def doChoiceList(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = [-stoi[x] for x in xs]
    if printHere:
      print(losses)
    assert len(losses) == 2
    if losses[0] == losses[1]:
        return random.choice([0,1])
    else:
       return np.argmin(losses)


with open("/checkpoint/mbaroni/char-rnn-exchange/candidate_adv_aoadj_testset.txt", "r") as inFile:
    dataset = [tuple(x.split("\t")) for x in inFile.read().strip().split("\n")]

choiceMasc = [0,0]
choiceFem = [0,0]

for adverb, adjective in dataset:
    adjectiveA = adjective[:-1]+"a"
    if adverb not in stoi or adjective not in stoi or adjectiveA not in stoi:
        continue
    masculine = [adjective, adjectiveA]
    choiceMasc[doChoiceList(masculine)] += 1
    print(choiceMasc[0] / sum(choiceMasc))
    feminine = [adjectiveA, adjective]
    choiceFem[doChoiceList(feminine)] += 1
    print(choiceFem[0] / sum(choiceFem))


with open("/checkpoint/mbaroni/char-rnn-exchange/candidate_adv_aeadj_testset.txt", "r") as inFile:
   dataset = [tuple(x.split("\t")) for x in inFile.read().strip().split("\n")]

choiceSg = [0,0]
choicePl = [0,0]

for adverb, adjective in dataset:
    adjectiveE = adjective[:-1]+"e"
    if adverb not in stoi or adverb not in stoi or adjectiveE not in stoi:
      continue
    singular = [adjective, adjectiveE]
    choiceSg[doChoiceList(singular)] += 1
    print(choiceSg[0] / sum(choiceSg))
    plural = [adjectiveE, adjective]
    choicePl[doChoiceList(plural)] += 1
    print(choicePl[0] / sum(choicePl))




with open("/checkpoint/mbaroni/char-rnn-exchange/candidate_eadj_aonoun_testset.txt", "r") as inFile:
   dataset = [tuple(x.split(" ")) for x in inFile.read().strip().split("\n")]

choice2Masc = [0,0]
choice2Fem = [0,0]

for adjective, noun in dataset:
    nounA = noun[:-1]+"a"
    if adjective not in stoi or noun not in stoi or nounA not in stoi:
       continue
    masc = ["il", "la"]
    choice2Masc[doChoiceList(masc)] += 1
    print(choice2Masc[0] / sum(choice2Masc))
    fem = ["il", "la"]
    choice2Fem[doChoiceList(fem)] += 1
    print(choice2Fem[1] / sum(choice2Fem))


print("/checkpoint/mbaroni/char-rnn-exchange/candidate_adv_aoadj_testset.txt")
print(choiceMasc[0] / sum(choiceMasc))
print(choiceFem[0] / sum(choiceFem))
print("/checkpoint/mbaroni/char-rnn-exchange/candidate_adv_aeadj_testset.txt")
print(choiceSg[0] / sum(choiceSg))
print(choicePl[0] / sum(choicePl))
print("/checkpoint/mbaroni/char-rnn-exchange/candidate_eadj_aonoun_testset.txt")
print(choice2Masc[0] / sum(choice2Masc))
print(choice2Fem[1] / sum(choice2Fem))




