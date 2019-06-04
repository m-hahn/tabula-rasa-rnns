assert False, "Obsolete"

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





def plusL(its):
  for it in its:
       for x in it:
           yield x

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


import random




from corpusIterator import CorpusIterator


plurals = set()

training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True)

for sentence in training.iterator():
 for line in sentence:
   if line["posUni"] == "NOUN":
      morph = line["morph"]
      if "Number=Plur" in  morph and "Case=Dat" not in morph:
        if "|" not in line["lemma"] and line["lemma"].lower() != line["word"]:
          plurals.add((line["lemma"].lower(), line["word"]))

formations = {"e" : set(), "n" : set(), "s" : set(), "same" : set(), "r" : set()}

for singular, plural in plurals:
  if len(singular) == len(plural):
    if singular[-1] == plural[-1]:
      formations["same"].add((singular, plural))
    else:
       print((singular, plural))
  elif plural.endswith("n"):
     formations["n"].add((singular, plural))
  elif plural.endswith("s"):
     formations["s"].add((singular, plural))
  elif plural.endswith("e"):
     formations["e"].add((singular, plural))
  elif plural.endswith("r"):
     formations["r"].add((singular, plural))
  else:
      print((singular, plural))

#print(formations["n"])
#print(formations["same"])


print({x:len(y) for x, y in formations.items()})

